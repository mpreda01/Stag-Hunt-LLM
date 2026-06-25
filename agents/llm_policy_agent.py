"""
agents/llm_policy_agent.py  —  LoRA + PPO fine-tuning of Qwen3-4B

Architecture
------------
    obs -> text prompt
        -> Qwen3-4B + LoRA adapters  (partially trainable)
        -> generate action token ("LEFT" / "DOWN" / "RIGHT" / "UP")
        -> parse action int
        -> env.step()
        -> team reward -> PPO update on LoRA weights

What is trained
---------------
    Only LoRA adapter weights on q_proj and v_proj of every transformer
    layer.  The base Qwen3-4B weights are frozen in 4-bit NF4.
    Typical LoRA param count for Qwen3-4B at rank 16: ~10-20M params,
    vs 4B total — about 0.5% of the model.

Why LoRA + PPO instead of the previous approaches
--------------------------------------------------
    - Hidden-state REINFORCE: embeddings were identical across all states
      (L2 dist = 0.0), so gradients were zero and nothing learned.
    - Token-logit REINFORCE: the single temperature scalar moved only
      0.001 after 90 episodes — the LLM's next-token distribution after
      "ACTION:" was essentially state-invariant even at the logit level.
    - LoRA + PPO: gradients flow INTO the model weights directly, so the
      model can actually change which action it predicts for different
      coordinate values.  The L40 (48 GB VRAM) makes this feasible.

PPO vs REINFORCE
----------------
    PPO uses a clipped surrogate objective that prevents destructively
    large weight updates, which is critical when fine-tuning a pretrained
    LLM — a single bad REINFORCE update can collapse the language model.
    The clip ratio (epsilon=0.2) keeps updates stable.

    PPO also uses a value network (critic) to estimate baselines, which
    dramatically reduces gradient variance compared to plain REINFORCE.

Memory budget on L40 (48 GB)
------------------------------
    - Qwen3-4B in 4-bit NF4:              ~3.5 GB
    - LoRA adapter weights (fp32):         ~0.2 GB
    - Optimizer states (AdamW, fp32):      ~0.4 GB
    - PPO rollout buffer (200 steps x 2):  ~1.0 GB
    - Activations during backward:         ~8-12 GB
    Total estimated:                       ~15 GB  (well within 48 GB)
"""

import re
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_WORDS  = ["LEFT", "DOWN", "RIGHT", "UP"]
ACTION_TO_INT = {w: i for i, w in enumerate(ACTION_WORDS)}
INT_TO_ACTION = {i: w for i, w in enumerate(ACTION_WORDS)}
DEFAULT_MODEL = "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# 1.  Qwen3-4B + LoRA  (the policy network)
# ---------------------------------------------------------------------------

class QwenLoRAPolicy:
    """
    Qwen3-4B loaded in 4-bit NF4 with LoRA adapters on attention projections.

    generate_action(prompt) -> (action_int, log_prob, entropy, response_text)
        Runs a constrained generation: forces the model to produce one of
        the four action words immediately after "ACTION:" by sampling from
        the action-token logits only.  This gives us log_prob for PPO.

    The value head is a small MLP on top of the last hidden state,
    used as the PPO critic baseline.
    """

    def __init__(
        self,
        model_name: str  = DEFAULT_MODEL,
        device:     str  = "cuda",
        lora_rank:  int  = 16,
        lora_alpha: int  = 32,
        lora_dropout: float = 0.05,
    ):
        self.device     = device
        self.model_name = model_name

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,   # bfloat16 for stability
            bnb_4bit_use_double_quant = True,              # saves another ~0.4 GB
        )

        print(f"[QwenLoRAPolicy] Loading {model_name} in 4-bit NF4 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map          = device,
            torch_dtype         = torch.bfloat16,
        )

        # LoRA config — target the attention q and v projections
        lora_config = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            lora_dropout   = lora_dropout,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
            target_modules = ["q_proj", "v_proj"],
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # Resolve action token ids — must be single tokens
        self.action_token_ids = []
        for word in ACTION_WORDS:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            assert len(ids) == 1, (
                f"'{word}' tokenises to {len(ids)} tokens. "
                f"Each action word must be a single token."
            )
            self.action_token_ids.append(ids[0])
        print(f"[QwenLoRAPolicy] Action token ids: "
              f"{ {w: i for w, i in zip(ACTION_WORDS, self.action_token_ids)} }")

        # Value head (critic) — small MLP on last hidden state
        hidden_dim = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        print(f"[QwenLoRAPolicy] Ready. hidden_dim={hidden_dim}")

    def forward_action_and_value(
        self,
        prompt: str,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass that returns both action and value estimate.

        Returns:
            action_int  : int in {0,1,2,3}
            log_prob    : scalar Tensor  (differentiable through LoRA)
            entropy     : scalar Tensor  (action distribution entropy)
            value       : scalar Tensor  (critic estimate)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states = True,
        )

        # --- Action distribution from action-token logits ---
        last_logits = outputs.logits[0, -1, :]          # (vocab_size,)
        action_logits = torch.stack([
            last_logits[tid] for tid in self.action_token_ids
        ])                                               # (4,)
        action_probs  = torch.softmax(action_logits, dim=0)
        dist          = torch.distributions.Categorical(action_probs)
        action_tensor = dist.sample()
        log_prob      = dist.log_prob(action_tensor)
        entropy       = dist.entropy()
        action_int    = int(action_tensor.item())

        # --- Value estimate from last hidden state ---
        last_hidden = outputs.hidden_states[-1][0, -1, :]   # (hidden_dim,)
        value       = self.value_head(last_hidden.float()).squeeze()

        return action_int, log_prob, entropy, value

    @torch.no_grad()
    def get_action_greedy(self, prompt: str) -> int:
        """Greedy action for evaluation (no gradient, no sampling)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        ).to(self.device)
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        action_logits = torch.stack([
            last_logits[tid] for tid in self.action_token_ids
        ])
        return int(action_logits.argmax().item())

    def parameters(self):
        """All trainable parameters: LoRA adapters + value head."""
        return list(self.model.parameters()) + list(self.value_head.parameters())

    def save(self, path: str):
        """Save LoRA adapters and value head."""
        self.model.save_pretrained(path + "_lora")
        torch.save(self.value_head.state_dict(), path + "_value_head.pt")
        print(f"[QwenLoRAPolicy] Saved -> {path}_lora  +  {path}_value_head.pt")

    def load(self, path: str):
        """Load LoRA adapters and value head."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path + "_lora")
        self.value_head.load_state_dict(
            torch.load(path + "_value_head.pt", map_location=self.device)
        )
        print(f"[QwenLoRAPolicy] Loaded <- {path}_lora")


# ---------------------------------------------------------------------------
# 2.  PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    One PPO agent wrapping a shared QwenLoRAPolicy.

    Both Agent A and Agent B share the same QwenLoRAPolicy instance
    (same LoRA weights) but maintain independent rollout buffers.
    This doubles the effective batch size and is valid because the prompt
    already encodes the agent identity ("You are Agent A/B").

    PPO update (called at end of each episode):
        1. Compute discounted returns G_t
        2. Compute advantages A_t = G_t - V(s_t)
        3. PPO clipped surrogate loss:
               L_clip = E[ min(r_t * A_t,  clip(r_t, 1-e, 1+e) * A_t) ]
           where r_t = pi(a_t|s_t) / pi_old(a_t|s_t)
        4. Value loss:  L_vf = E[ (V(s_t) - G_t)^2 ]
        5. Entropy bonus: L_ent = E[ H(pi(s_t)) ]
        6. Total loss = -L_clip + c_vf * L_vf - c_ent * L_ent
    """

    def __init__(
        self,
        policy:     QwenLoRAPolicy,
        agent_id:   str,
        lr:         float = 1e-4,
        gamma:      float = 0.99,
        clip_eps:   float = 0.2,
        vf_coeff:   float = 0.5,
        ent_coeff:  float = 0.01,
    ):
        self.policy    = policy
        self.agent_id  = agent_id
        self.gamma     = gamma
        self.clip_eps  = clip_eps
        self.vf_coeff  = vf_coeff
        self.ent_coeff = ent_coeff

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        # Per-episode rollout buffers
        self._log_probs_old : list[torch.Tensor] = []
        self._log_probs_new : list[torch.Tensor] = []
        self._values        : list[torch.Tensor] = []
        self._entropies     : list[torch.Tensor] = []
        self._rewards       : list[float]        = []

        # Logging
        self.loss_history:           list[float] = []
        self.episode_return_history: list[float] = []

    def select_action(
        self,
        prompt:  str,
        greedy:  bool = False,
    ) -> int:
        """
        Training: full forward pass, stores log_prob, value, entropy.
        Eval:     greedy argmax, no storage.
        """
        if greedy:
            return self.policy.get_action_greedy(prompt)

        action_int, log_prob, entropy, value = \
            self.policy.forward_action_and_value(prompt)

        # Store old log_prob (detached) for PPO ratio computation
        self._log_probs_old.append(log_prob.detach())
        # Store new log_prob (with grad) for the surrogate loss
        self._log_probs_new.append(log_prob)
        self._values.append(value)
        self._entropies.append(entropy)

        return action_int

    def store_reward(self, reward: float):
        self._rewards.append(reward)

    def update(self) -> float:
        """PPO update at end of episode. Returns total loss value."""
        T = len(self._rewards)
        if T == 0:
            return 0.0

        # --- Discounted returns ---
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32,
                                 device=self.policy.device)

        # --- Advantages ---
        values_t  = torch.stack(self._values)             # (T,)
        adv_t     = returns_t - values_t.detach()

        # Normalise advantages
        if adv_t.std().item() > 1e-6:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # --- PPO clipped surrogate ---
        log_probs_old = torch.stack(self._log_probs_old)  # (T,) detached
        log_probs_new = torch.stack(self._log_probs_new)  # (T,) with grad
        entropies     = torch.stack(self._entropies)       # (T,)

        ratio      = torch.exp(log_probs_new - log_probs_old)
        surr1      = ratio * adv_t
        surr2      = torch.clamp(ratio, 1 - self.clip_eps,
                                        1 + self.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- Value loss ---
        value_loss  = ((values_t - returns_t) ** 2).mean()

        # --- Entropy bonus ---
        entropy_loss = -entropies.mean()

        # --- Total loss ---
        loss = (policy_loss
                + self.vf_coeff  * value_loss
                + self.ent_coeff * entropy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients — important for LLM fine-tuning stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.episode_return_history.append(sum(self._rewards))

        # Clear buffers
        self._log_probs_old.clear()
        self._log_probs_new.clear()
        self._values.clear()
        self._entropies.clear()
        self._rewards.clear()

        return loss_val

    def save(self, path: str):
        torch.save({
            "optimizer":              self.optimizer.state_dict(),
            "loss_history":           self.loss_history,
            "episode_return_history": self.episode_return_history,
        }, path + "_optim.pt")
        print(f"[Agent {self.agent_id}] Optimizer saved -> {path}_optim.pt")

    def load(self, path: str):
        ckpt = torch.load(path + "_optim.pt", map_location="cpu")
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.loss_history           = ckpt.get("loss_history", [])
        self.episode_return_history = ckpt.get("episode_return_history", [])
        print(f"[Agent {self.agent_id}] Optimizer loaded <- {path}_optim.pt")
