"""
qwen3b.py  —  Qwen2.5-3B-Instruct + LoRA policy for Stag Hunt

Components
----------
    1. Model loading (4-bit NF4 + LoRA on attn projections)
    2. Prompt generation with Manhattan distance / direction reasoning
    3. Action parser  (<action>...</action> tags → int)
    4. Log-probability extraction for policy gradient loss
"""

import re
import random
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

ACTION_MAP: dict[str, int] = {
    "UP":    0,
    "DOWN":  1,
    "LEFT":  2,
    "RIGHT": 3,
    "STAY":  4,
}
INT_TO_ACTION: dict[int, str] = {v: k for k, v in ACTION_MAP.items()}
N_ACTIONS = len(ACTION_MAP)

# Action tokens that the LLM will generate inside <action>…</action>
# We match only these, then fall back to random.
_ACTION_PATTERN = re.compile(
    r"<action>\s*(UP|DOWN|LEFT|RIGHT|STAY)\s*</action>",
    re.IGNORECASE,
)

# Direction helpers
_DX_LABEL = {-1: "West",  0: "",  1: "East"}
_DY_LABEL = {-1: "North", 0: "",  1: "South"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manhattan_direction(src: tuple[int, int], dst: tuple[int, int]) -> str:
    """
    Returns a human-readable direction + distance string, e.g.:
        '3 steps South and 1 step East'
        '2 steps North'
        'same cell'
    """
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]

    if dx == 0 and dy == 0:
        return "same cell"

    parts: list[str] = []
    if dy != 0:
        label = _DY_LABEL[1 if dy > 0 else -1]
        parts.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} {label}")
    if dx != 0:
        label = _DX_LABEL[1 if dx > 0 else -1]
        parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} {label}")

    return " and ".join(parts)


# ---------------------------------------------------------------------------
# 1. Prompt generation
# ---------------------------------------------------------------------------

def generate_stag_hunt_prompt(
    agent_pos:       tuple[int, int],
    teammate_pos:    tuple[int, int],
    stag_pos:        tuple[int, int],
    hares_positions: list[tuple[int, int]],
    grid_size:       int,
    history_log:     list[str],
) -> str:
    """
    Build a Chain-of-Thought prompt that includes Manhattan distances and
    direction vectors so the model can reason spatially without parsing raw
    coordinates.

    Parameters
    ----------
    agent_pos       : (x, y) of this agent; (0,0) is top-left.
    teammate_pos    : (x, y) of the cooperating partner.
    stag_pos        : (x, y) of the stag.
    hares_positions : list of (x, y) for each hare.
    grid_size       : side length of the square grid (N×N).
    history_log     : last few "<AgentX>: ACTION" strings for context.

    Returns
    -------
    str  —  full prompt ready to feed to the tokenizer.
    """
    stag_rel   = _manhattan_direction(agent_pos, stag_pos)
    team_rel   = _manhattan_direction(agent_pos, teammate_pos)

    hare_descs: list[str] = []
    for i, hp in enumerate(hares_positions):
        rel = _manhattan_direction(agent_pos, hp)
        hare_descs.append(f"  Hare {i+1}: {hp}  (relative: {rel})")
    hares_str = "\n".join(hare_descs) if hare_descs else "  None visible"

    history_str = (
        "\n".join(f"  {h}" for h in history_log[-5:])
        if history_log
        else "  No history yet."
    )

    prompt = (
        f"You are an AI Hunter playing the Stag Hunt game on a "
        f"{grid_size}×{grid_size} grid. Coordinates are (col, row) "
        f"with (0,0) at the top-left corner.\n\n"
        f"Your current position : {agent_pos}\n"
        f"Teammate position     : {teammate_pos}  (relative: {team_rel})\n"
        f"Stag position         : {stag_pos}  (relative: {stag_rel})\n"
        f"Hares:\n{hares_str}\n\n"
        f"Recent action history:\n{history_str}\n\n"
        f"Rules:\n"
        f"  • Catching the Stag with your teammate simultaneously → +5 each.\n"
        f"  • Stepping on the Stag alone               → -5 (mauled).\n"
        f"  • Stepping on a Hare alone                 → +1 (safe).\n"
        f"  • The Stag moves toward the nearest agent each turn.\n\n"
        f"Instructions:\n"
        f"  Think step-by-step about your teammate's likely move and whether "
        f"you can both reach the Stag on the same turn. Express your full "
        f"reasoning inside <think>…</think> tags. Then output exactly one "
        f"action word (UP / DOWN / LEFT / RIGHT / STAY) inside "
        f"<action>…</action> tags.\n\n"
        f"<think>"
    )
    return prompt


# ---------------------------------------------------------------------------
# 2. Output parser
# ---------------------------------------------------------------------------

def parse_llm_output(output_text: str) -> int:
    """
    Extract the action integer from the model's raw text output.

    Tries (in order):
      1. <action>WORD</action> pattern (canonical)
      2. Bare keyword anywhere in the cleaned text after </think>
      3. Random fallback

    Returns
    -------
    int in [0, N_ACTIONS)
    """
    # Strip the <think> block — we only care about the action declaration
    post_think = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)

    # Primary: tagged action
    m = _ACTION_PATTERN.search(post_think)
    if m:
        return ACTION_MAP[m.group(1).upper()]

    # Secondary: bare keyword (model forgot tags but said the word)
    clean = post_think.upper()
    for word, idx in ACTION_MAP.items():
        if re.search(rf"\b{word}\b", clean):
            return idx

    # Fallback
    fallback = random.randint(0, N_ACTIONS - 1)
    print(
        f"[parse_llm_output] Could not parse action. "
        f"Falling back to random ({INT_TO_ACTION[fallback]}). "
        f"Raw text (first 200 chars): {output_text[:200]!r}"
    )
    return fallback


# ---------------------------------------------------------------------------
# 3. Model wrapper
# ---------------------------------------------------------------------------

class QwenStagHuntPolicy:
    """
    Wraps Qwen2.5-3B-Instruct with LoRA adapters for policy gradient training.

    The base weights are frozen (via 4-bit NF4 quantisation); only the LoRA
    delta matrices are trainable.  This allows training on an RTX 2080 (11 GB)
    without offloading, and on an L40 (48 GB) with headroom for larger batches.

    Public interface
    ----------------
    generate_action(prompt)       → (action_int, raw_text)
    log_probs_of_response(prompt, response) → Tensor (scalar)  ← for PG loss
    save(path) / load(path)
    """

    def __init__(
        self,
        model_name:  str = MODEL_NAME,
        device:      str = "cuda",
        lora_rank:   int = 16,
        lora_alpha:  int = 32,
        lora_dropout: float = 0.05,
        max_new_tokens: int = 256,
    ):
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self.model_name     = model_name

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
        )

        print(f"[QwenStagHuntPolicy] Loading {model_name} in 4-bit NF4 …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side = "left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map          = device,
        )

        lora_config = LoraConfig(
            task_type    = TaskType.CAUSAL_LM,
            r            = lora_rank,
            lora_alpha   = lora_alpha,
            lora_dropout = lora_dropout,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            bias         = "none",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        print(f"[QwenStagHuntPolicy] Ready on {device}.")

    # ------------------------------------------------------------------
    # Action generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_action(
        self,
        prompt: str,
        temperature: float = 0.7,
        do_sample:   bool  = True,
    ) -> tuple[int, str]:
        """
        Generate one response token-sequence and parse the action.

        Returns
        -------
        (action_int, raw_generated_text)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 1024,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens = self.max_new_tokens,
            do_sample      = do_sample,
            temperature    = temperature,
            pad_token_id   = self.tokenizer.pad_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        new_ids  = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return parse_llm_output(raw_text), raw_text

    # ------------------------------------------------------------------
    # Log-probability of a (prompt, response) pair — used for PG loss
    # ------------------------------------------------------------------

    def log_probs_of_response(
        self,
        prompt:   str,
        response: str,
    ) -> torch.Tensor:
        """
        Forward-pass the concatenated (prompt + response) through the model and
        return the sum of log-probabilities over the *response* tokens.

        This gives log π_θ(a | s), the quantity needed for REINFORCE /
        GRPO policy gradient:

            loss = -log_prob * advantage

        Returns
        -------
        Tensor (scalar, requires_grad=True)
        """
        full_text = prompt + response

        full_enc   = self.tokenizer(
            full_text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 2048,
        ).to(self.device)

        prompt_enc = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 2048,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.model(
                input_ids      = full_enc["input_ids"],
                attention_mask = full_enc["attention_mask"],
            )

        # logits: (1, seq_len, vocab_size)
        logits = outputs.logits[0]                    # (seq_len, vocab_size)
        log_p  = torch.log_softmax(logits, dim=-1)   # (seq_len, vocab_size)

        # Shift: logits[t] predicts token[t+1]
        # We want the log-prob assigned to each *response* token.
        target_ids = full_enc["input_ids"][0]         # (seq_len,)

        # Response token indices in the full sequence: [prompt_len, seq_len)
        response_start = prompt_len          # first response token position
        response_end   = target_ids.shape[0] # exclusive

        if response_start >= response_end:
            # Edge case: response was empty or got truncated
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # log_p[t] = log P(token[t+1] | token[0..t])
        # So for response token at position i (>= response_start),
        # we use log_p[i-1, target_ids[i]]
        positions   = torch.arange(response_start, response_end, device=self.device)
        token_logps = log_p[positions - 1, target_ids[positions]]  # (response_len,)

        return token_logps.sum()

    # ------------------------------------------------------------------
    # Reference model KL helper  (optional, for KL-penalised GRPO)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reference_log_probs(
        self,
        prompt:   str,
        response: str,
    ) -> torch.Tensor:
        """
        Compute log π_ref(response | prompt) using the *base* (non-LoRA) weights.

        We temporarily disable the LoRA adapters, run the forward pass,
        and re-enable them.  This is cheap because no gradient is needed.
        """
        self.model.disable_adapter_layers()
        ref_lp = self.log_probs_of_response(prompt, response).detach()
        self.model.enable_adapter_layers()
        return ref_lp

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save LoRA adapter weights only (much smaller than full model)."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[QwenStagHuntPolicy] Saved LoRA adapters → {path}")

    def load(self, path: str):
        """Load LoRA adapter weights from a previous save."""
        from peft import PeftModel
        self.model.load_adapter(path, adapter_name="default")
        print(f"[QwenStagHuntPolicy] Loaded LoRA adapters ← {path}")
