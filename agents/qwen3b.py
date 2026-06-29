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

from utils.const import ACTION_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


INT_TO_ACTION: dict[int, str] = {v: k for k, v in ACTION_MAP.items()}
N_ACTIONS = len(ACTION_MAP)

# Action tokens that the LLM will generate inside <action>…</action>
# We match only these, then fall back to random.
_ACTION_PATTERN = re.compile(
    r"<action>\s*(UP|DOWN|LEFT|RIGHT)\s*</action>",
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
    history_log:     list[str],          # kept for API compatibility, unused
) -> str:
    """
    Minimal prompt: raw coordinates + action choices only (~25 tokens).
    GRPO tunes the action selection — no prompt engineering needed.
    """
    hares_str = " ".join(f"H:{h}" for h in hares_positions)
    return (
        f"A:{agent_pos} T:{teammate_pos} S:{stag_pos} {hares_str} G:{grid_size}\n"
        f"Action (UP/DOWN/LEFT/RIGHT):"
    )


# ---------------------------------------------------------------------------
# 2. Output parser
# ---------------------------------------------------------------------------

def parse_llm_output(output_text: str) -> int:
    """
    Extract the action integer from the model's raw text output.

    Tries (in order):
      1. <action>WORD</action> tag anywhere in the text (canonical)
      2. Last non-empty line is exactly one action keyword
      3. Last occurrence of any action keyword in the full text
      4. Random fallback

    Returns
    -------
    int in [0, N_ACTIONS)
    """
    # Primary: tagged action anywhere in the full response
    m = _ACTION_PATTERN.search(output_text)
    if m:
        return ACTION_MAP[m.group(1).upper()]

    # Secondary: last non-empty line contains exactly one action keyword
    lines = [l.strip() for l in output_text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].upper()
        for word, idx in ACTION_MAP.items():
            if re.fullmatch(rf"[\W]*{word}[\W]*", last):
                return idx

    # Tertiary: last occurrence of any action keyword in the full text
    # (model often restates its decision near the end of its reasoning)
    best_pos, best_idx = -1, -1
    upper = output_text.upper()
    for word, idx in ACTION_MAP.items():
        for m2 in re.finditer(rf"\b{word}\b", upper):
            if m2.start() > best_pos:
                best_pos, best_idx = m2.start(), idx
    if best_idx >= 0:
        return best_idx

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
        max_new_tokens: int = 8,
    ):
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self.model_name     = model_name

        # Prevent tokenizer from spawning threads that can deadlock on SLURM
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
        )

        print(f"[QwenStagHuntPolicy] Loading tokenizer for {model_name} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side  = "left",
            use_fast      = False,   # fast tokenizer uses Rust threads — avoid on SLURM
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("[QwenStagHuntPolicy] Tokenizer loaded.", flush=True)

        print(f"[QwenStagHuntPolicy] Loading model weights in 4-bit NF4 …", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map          = {"": device},
            dtype               = torch.float16,
            low_cpu_mem_usage   = True,
        )
        print("[QwenStagHuntPolicy] Base model loaded.", flush=True)

        print("[QwenStagHuntPolicy] Applying LoRA adapters …", flush=True)
        lora_config = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            lora_dropout   = lora_dropout,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            bias           = "none",
        )
        self.model = get_peft_model(base_model, lora_config)
        print("[QwenStagHuntPolicy] LoRA adapters applied.", flush=True)
        self.model.print_trainable_parameters()
        print(f"[QwenStagHuntPolicy] Ready on {device}.", flush=True)

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
