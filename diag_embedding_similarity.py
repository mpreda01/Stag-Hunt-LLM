"""
diag_embedding_similarity.py

Diagnostic: checks whether the LLM encoder produces meaningfully different
hidden states for different observations, using the REAL two-shot prompts
that the network sees during training.

Tests Agent A's embedding across:
    1. Two consecutive states (small positional delta — hardest case)
    2. Two very different states (agents far apart vs. adjacent to stag)
    3. Same state twice (sanity check — should give similarity ≈ 1.0)

Also checks whether the initial (untrained) PolicyHead outputs a near-uniform
distribution, which would confirm the flat-loss hypothesis.

Usage:
    python diag_embedding_similarity.py

Requires the project to be on PYTHONPATH (run from the project root).
"""

import torch
import torch.nn.functional as F
import numpy as np

from agents.qwen4b import obs_to_prompt
from agents.llm_policy_agent import LLMEncoder, PolicyHead


# ---------------------------------------------------------------------------
# Define a handful of realistic observations for Agent A
# obs layout (flat, 10 values): ax, ay, bx, by, sx, sy, p1x, p1y, p2x, p2y
# ---------------------------------------------------------------------------

# State 0: starting position, stag in centre, plants in corners
STATE_0 = np.array([0, 0,  4, 0,  2, 2,  1, 3,  3, 1], dtype=float)

# State 1: one step right for A, stag moved slightly — consecutive to state 0
STATE_1 = np.array([1, 0,  4, 0,  2, 2,  1, 3,  3, 1], dtype=float)

# State 2: A adjacent to stag, B far away — very different strategic situation
STATE_2 = np.array([2, 2,  0, 0,  2, 2,  1, 3,  3, 1], dtype=float)

# State 3: both agents adjacent to stag — cooperative catch imminent
STATE_3 = np.array([2, 1,  2, 3,  2, 2,  1, 3,  3, 1], dtype=float)

STATES = {
    "state_0 (start, A at corner)":            STATE_0,
    "state_1 (A one step right, consecutive)":  STATE_1,
    "state_2 (A on stag, B far)":              STATE_2,
    "state_3 (both adjacent stag)":            STATE_3,
}

PROMPT_TYPE = "4"   # two-shot — matches training config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 65)

    # Load encoder (frozen Qwen)
    encoder = LLMEncoder(device=device)
    head    = PolicyHead(hidden_dim=encoder.hidden_dim)

    # ---------------------------------------------------------------------------
    # 1. Encode all states and print the prompts so we can visually verify them
    # ---------------------------------------------------------------------------
    print("\n[PROMPTS] Showing Agent A prompt for each state:\n")
    embeddings = {}
    for name, state in STATES.items():
        # obs_to_prompt expects a (2, 10) multiagent obs; we stack a dummy B obs
        multiagent_obs = np.stack([state, state])   # B's view doesn't matter here
        prompt_a, _ = obs_to_prompt(multiagent_obs, prompot_type=PROMPT_TYPE)

        print(f"--- {name} ---")
        print(prompt_a)
        print()

        h = encoder.encode(prompt_a)
        embeddings[name] = h
        print(f"  embedding norm : {h.norm().item():.4f}")
        print(f"  embedding mean : {h.mean().item():.6f}")
        print(f"  embedding std  : {h.std().item():.6f}")
        print()

    # ---------------------------------------------------------------------------
    # 2. Pairwise cosine similarity and L2 distance
    # ---------------------------------------------------------------------------
    names = list(embeddings.keys())
    print("=" * 65)
    print("[SIMILARITY] Pairwise cosine similarity between Agent A embeddings:")
    print("  (closer to 1.0 = embeddings are nearly identical)\n")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n_i, n_j = names[i], names[j]
            h_i, h_j = embeddings[n_i], embeddings[n_j]
            cos  = cosine_sim(h_i, h_j)
            l2   = l2_dist(h_i, h_j)
            flag = "  ⚠ NEARLY IDENTICAL" if cos > 0.999 else ""
            print(f"  {n_i[:35]:35s}")
            print(f"  vs {n_j[:35]:35s}")
            print(f"    cosine sim = {cos:.6f} | L2 dist = {l2:.4f}{flag}")
            print()

    # ---------------------------------------------------------------------------
    # 3. Sanity check: same state twice → should be exactly 1.0
    # ---------------------------------------------------------------------------
    print("=" * 65)
    print("[SANITY] Same state encoded twice (expect cosine ≈ 1.0000):")
    multiagent_obs = np.stack([STATE_0, STATE_0])
    prompt_a, _ = obs_to_prompt(multiagent_obs, prompot_type=PROMPT_TYPE)
    h_a = encoder.encode(prompt_a)
    h_b = encoder.encode(prompt_a)
    print(f"  cosine sim = {cosine_sim(h_a, h_b):.6f}")
    print()

    # ---------------------------------------------------------------------------
    # 4. PolicyHead output distribution (untrained)
    #    If all outputs are ~0.25 the head starts uniform — expected.
    #    What matters is whether DIFFERENT embeddings produce DIFFERENT outputs.
    # ---------------------------------------------------------------------------
    print("=" * 65)
    print("[POLICY HEAD] Action probability distribution per state (untrained head):")
    print("  (if all rows are ~[0.25, 0.25, 0.25, 0.25] → head not differentiating)\n")
    action_labels = ["LEFT", "DOWN", "RIGHT", "UP"]
    for name, h in embeddings.items():
        probs = head(h.unsqueeze(0)).detach().squeeze()
        prob_str = "  ".join(f"{a}={p:.4f}" for a, p in zip(action_labels, probs))
        print(f"  {name[:45]:45s}  {prob_str}")

    # ---------------------------------------------------------------------------
    # 5. Verdict
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[VERDICT]")

    sims = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sims.append(cosine_sim(embeddings[names[i]], embeddings[names[j]]))

    avg_sim = sum(sims) / len(sims)
    max_sim = max(sims)
    min_sim = min(sims)

    print(f"  avg cosine similarity across all pairs : {avg_sim:.6f}")
    print(f"  max cosine similarity                  : {max_sim:.6f}")
    print(f"  min cosine similarity                  : {min_sim:.6f}")
    print()

    if max_sim > 0.999:
        print("  ✗ PROBLEM: at least one pair of different states produces nearly")
        print("    identical embeddings. The LLM is not discriminating between them.")
        print("    The PolicyHead cannot learn — consider a different encoding strategy.")
    elif avg_sim > 0.99:
        print("  ⚠ WARNING: embeddings are very similar on average (avg > 0.99).")
        print("    The PolicyHead has very little signal to work with.")
        print("    Learning will be extremely slow if it happens at all.")
    else:
        print("  ✓ OK: embeddings vary meaningfully across states.")
        print("    The PolicyHead has a real gradient signal to work with.")
        print("    If loss is still flat, the issue is in the training loop, not the encoder.")

    print("=" * 65)


if __name__ == "__main__":
    main()
