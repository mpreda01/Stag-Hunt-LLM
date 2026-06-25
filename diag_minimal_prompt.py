"""
diag_minimal_prompt.py

Diagnostic for the minimal-prompt LLM encoder approach.

Checks:
    1. Print the actual prompts so you can visually verify the format
    2. Embedding variance — L2 dist and cosine sim between different states
    3. Embedding sanity — same state twice must give identical embedding
    4. PolicyHead sensitivity — different embeddings must give different outputs
    5. Gradient flow — one fake REINFORCE update must change the head weights
    6. Verdict — pass/fail with clear diagnosis

Run on the HPC (model already cached):
    HF_HOME=/scratch.hpc/matteo.preda/hf_cache python diag_minimal_prompt.py

Run locally (will try to download the model — only do this if you have it cached):
    python diag_minimal_prompt.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

from agents.llm_policy_agent import LLMEncoder, PolicyHead, build_minimal_prompt


# ---------------------------------------------------------------------------
# Test states
# obs layout: ax, ay, bx, by, sx, sy, p1x, p1y, p2x, p2y
# ---------------------------------------------------------------------------

STATES = {
    "t=0  A:(0,0) B:(4,0) S:(2,2)  [start]":
        np.array([0,0, 4,0, 2,2, 1,3, 3,1], dtype=float),

    "t=1  A:(1,0) B:(4,0) S:(2,2)  [A one step right]":
        np.array([1,0, 4,0, 2,2, 1,3, 3,1], dtype=float),

    "t=2  A:(2,2) B:(4,0) S:(2,2)  [A ON stag, B far — maul risk]":
        np.array([2,2, 4,0, 2,2, 1,3, 3,1], dtype=float),

    "t=3  A:(2,1) B:(2,3) S:(2,2)  [both adjacent, catch imminent]":
        np.array([2,1, 2,3, 2,2, 1,3, 3,1], dtype=float),

    "t=X  A:(0,4) B:(4,4) S:(1,1)  [very different config]":
        np.array([0,4, 4,4, 1,1, 3,2, 2,3], dtype=float),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item()

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Load encoder
    # ------------------------------------------------------------------
    encoder = LLMEncoder(device=device)
    head    = PolicyHead(hidden_dim=encoder.hidden_dim)

    # ------------------------------------------------------------------
    # 1. Print prompts
    # ------------------------------------------------------------------
    section("1. PROMPTS — visual check")
    print("  Verifying minimal prompt format for Agent A:\n")
    for name, state in STATES.items():
        prompt = build_minimal_prompt(state, agent="A")
        n_tokens = len(encoder.tokenizer.encode(prompt))
        print(f"  [{name}]")
        print(f"    prompt   : {prompt}")
        print(f"    n_tokens : {n_tokens}")
        print()

    # ------------------------------------------------------------------
    # 2. Encode all states
    # ------------------------------------------------------------------
    section("2. EMBEDDINGS — norm, mean, std per state")
    print("  (std should differ across states if embedding varies)\n")

    embeddings = {}
    for name, state in STATES.items():
        prompt = build_minimal_prompt(state, agent="A")
        h = encoder.encode(state)          # uses build_minimal_prompt internally
        embeddings[name] = h
        print(f"  {name[:55]:55s}")
        print(f"    norm={h.norm().item():.4f}  "
              f"mean={h.mean().item():.6f}  "
              f"std={h.std().item():.6f}")
        print()

    # ------------------------------------------------------------------
    # 3. Pairwise L2 and cosine similarity
    # ------------------------------------------------------------------
    section("3. PAIRWISE SIMILARITY")
    print("  L2 dist > 0 and cosine sim < 1.0 means embeddings differ.\n"
          "  Previous approach: L2 = 0.0000 for ALL pairs (broken).\n"
          "  Target: L2 >> 0, cosine sim < 0.999\n")

    names = list(embeddings.keys())
    sims, dists = [], []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            hi, hj = embeddings[ni], embeddings[nj]
            cs = cosine_sim(hi, hj)
            l2 = l2_dist(hi, hj)
            sims.append(cs)
            dists.append(l2)

            flag = ""
            if l2 < 0.001:
                flag = "  ✗ STILL BROKEN — identical embeddings"
            elif cs > 0.999:
                flag = "  ⚠ very similar"
            else:
                flag = "  ✓"

            print(f"  {ni[:40]:40s}")
            print(f"  vs {nj[:40]:40s}")
            print(f"    cosine={cs:.6f}  L2={l2:.4f}{flag}")
            print()

    # ------------------------------------------------------------------
    # 4. Sanity check — same state twice
    # ------------------------------------------------------------------
    section("4. SANITY — same state encoded twice (expect L2 = 0.0)")
    state0 = list(STATES.values())[0]
    h1 = encoder.encode(state0)
    h2 = encoder.encode(state0)
    l2_same = l2_dist(h1, h2)
    cs_same = cosine_sim(h1, h2)
    flag = "✓" if l2_same < 1e-4 else "✗ NOT DETERMINISTIC"
    print(f"  L2={l2_same:.6f}  cosine={cs_same:.6f}  {flag}")

    # ------------------------------------------------------------------
    # 5. PolicyHead sensitivity
    # ------------------------------------------------------------------
    section("5. POLICY HEAD — action distribution per state")
    print("  Distributions should DIFFER across states.\n"
          "  If all rows identical → head is not sensitive to input.\n")

    action_labels = ["LEFT", "DOWN", "RIGHT", "UP"]
    policy_outputs = []
    for name, h in embeddings.items():
        probs = head(h.unsqueeze(0)).detach().squeeze()
        policy_outputs.append(probs)
        prob_str = "  ".join(f"{a}={p:.4f}" for a, p in zip(action_labels, probs))
        print(f"  {name[:50]:50s}  {prob_str}")

    # Check if all outputs are identical
    all_same = all(
        (policy_outputs[0] - p).abs().max().item() < 1e-6
        for p in policy_outputs[1:]
    )
    if all_same:
        print("\n  ✗ All policy outputs identical — head is insensitive to input")
    else:
        print("\n  ✓ Policy outputs differ across states")

    # ------------------------------------------------------------------
    # 6. Gradient flow test
    # ------------------------------------------------------------------
    section("6. GRADIENT FLOW — one fake REINFORCE update")
    print("  All PolicyHead parameters should show max_delta > 0.\n")

    head_before = {k: v.clone() for k, v in head.named_parameters()}
    optimizer   = torch.optim.Adam(head.parameters(), lr=1e-3)

    # Fake episode: 5 steps with alternating returns
    log_probs = []
    returns   = torch.tensor([2.0, -1.0, 3.0, -2.0, 1.0])
    for state in list(STATES.values())[:5]:
        h     = encoder.encode(state)
        probs = head(h.unsqueeze(0))
        dist  = torch.distributions.Categorical(probs)
        a     = dist.sample()
        log_probs.append(dist.log_prob(a))

    loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Loss value: {loss.item():.4f}\n")
    all_updated = True
    for name, param in head.named_parameters():
        delta   = (param - head_before[name]).abs().max().item()
        updated = delta > 1e-9
        if not updated:
            all_updated = False
        print(f"  {name:45s}  max_delta={delta:.2e}  "
              f"{'✓ updated' if updated else '✗ no change'}")

    # ------------------------------------------------------------------
    # 7. Verdict
    # ------------------------------------------------------------------
    section("7. VERDICT")

    avg_l2   = sum(dists) / len(dists) if dists else 0.0
    max_l2   = max(dists) if dists else 0.0
    min_l2   = min(dists) if dists else 0.0
    avg_cos  = sum(sims)  / len(sims)  if sims  else 1.0

    print(f"  avg L2 dist across pairs : {avg_l2:.4f}")
    print(f"  min L2 dist              : {min_l2:.4f}")
    print(f"  max L2 dist              : {max_l2:.4f}")
    print(f"  avg cosine similarity    : {avg_cos:.6f}")
    print(f"  gradients flow           : {'yes' if all_updated else 'NO'}")
    print()

    if min_l2 < 0.001:
        print("  ✗ FAIL: At least one state pair still gives L2 ≈ 0.")
        print("    The minimal prompt did NOT fix the embedding collapse.")
        print("    Check the tokenizer — coordinates may be tokenised")
        print("    differently than expected (e.g. '0,0' -> ['0', ',', '0']).")
        sys.exit(1)
    elif avg_cos > 0.999:
        print("  ⚠ WARNING: Embeddings differ (L2 > 0) but are very similar.")
        print("    The MLP may still struggle to differentiate states.")
        print("    Consider reducing prompt length further or adding")
        print("    a projection layer before the PolicyHead.")
    elif not all_updated:
        print("  ⚠ WARNING: Embeddings look good but gradients are not")
        print("    flowing through the PolicyHead. Check for detach() issues.")
    else:
        print("  ✓ PASS: Embeddings vary meaningfully across states.")
        print("    Gradients flow through PolicyHead.")
        print("    The minimal-prompt approach should be trainable.")
        print("    Submit the training job.")

    print("=" * 65)


if __name__ == "__main__":
    main()
