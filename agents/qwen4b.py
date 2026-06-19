import numpy as np
from ollama import chat
import re, random
from utils.const import ACTION_MAP, ZERO_SHOT, ONE_SHOT, TWO_SHOT



def query_llm(prompt: str, model: str = "qwen3:4b-q4_K_M", think: bool = False) -> tuple[int, str]:
    """
    Feed a prompt to the Qwen model and return the parsed action.
    
    Returns:
        (action_int, raw_response)
        action_int: 0-3 (LEFT/DOWN/RIGHT/UP), or random fallback if parsing fails
    """
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=think,
    )
    
    raw = response.message.content  # includes <tool_call> block if think=True
    # Parse the action: look for "ACTION: <DIR>" in the response
    action_int = parse_action(raw)
    
    return action_int, raw


def parse_action(text: str) -> int:
    """Extract action integer from LLM response. Returns random fallback if not found."""
    
    
    # Strip <think>...</think> block if present before searching
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    match = re.search(r"ACTION:\s*(LEFT|DOWN|RIGHT|UP)", clean, re.IGNORECASE)
    if match:
        return ACTION_MAP[match.group(1).upper()]
    
    # Fallback: scan anywhere in the full text
    for action_str, action_int in ACTION_MAP.items():
        if action_str in clean.upper():
            return action_int
    
    print(f"[WARN] Could not parse action from response, using random. Response was:\n{clean[:200]}")
    return random.randint(0, 3)

def obs_to_prompt(obs, prompot_type, grid_size=(5, 5)) -> str | tuple[str, str]:
    """
    Convert coords observation to prompt(s).
    - Single agent (obs shape (10,)): pass agent="A" or "B", returns one prompt.
    - Multiagent (obs shape (2,10)): returns (prompt_A, prompt_B) tuple, agent param ignored.
    """
    obs = np.array(obs)
    if prompot_type == "2":
        
        if obs.ndim == 2:
            # Multiagent: row 0 is A's view, row 1 is B's view
            return (
                ZERO_SHOT + build_prompt(obs[0], agent="A", grid_size=grid_size),
                ZERO_SHOT + build_prompt(obs[1], agent="B", grid_size=grid_size),
            )
        else:
            # Single agent
            return ZERO_SHOT + build_prompt(obs, agent="A", grid_size=grid_size)
    
    elif prompot_type == "3":
        
        if obs.ndim == 2:
            # Multiagent: row 0 is A's view, row 1 is B's view
            return (
                ONE_SHOT + build_prompt(obs[0], agent="A", grid_size=grid_size),
                ONE_SHOT + build_prompt(obs[1], agent="B", grid_size=grid_size),
            )
        else:
            # Single agent
            return ONE_SHOT + build_prompt(obs, agent="A", grid_size=grid_size)
    
    elif prompot_type == "4":
        
        if obs.ndim == 2:
            # Multiagent: row 0 is A's view, row 1 is B's view
            return (
                TWO_SHOT + build_prompt(obs[0], agent="A", grid_size=grid_size),
                TWO_SHOT + build_prompt(obs[1], agent="B", grid_size=grid_size),
            )
        else:
            # Single agent
            return TWO_SHOT + build_prompt(obs, agent="A", grid_size=grid_size)



def build_prompt(obs, agent: str, grid_size=(5, 5)) -> str:
    """Build a single agent prompt from a flat (10,) obs array."""
    ax, ay = int(obs[0]), int(obs[1])
    bx, by = int(obs[2]), int(obs[3])
    sx, sy = int(obs[4]), int(obs[5])
    plants = [(int(obs[i]), int(obs[i+1])) for i in range(6, len(obs), 2)]
    plants_str = ", ".join(f"({px},{py})" for px, py in plants)

    teammate = "B" if agent == "A" else "A"
    return f"""You are Agent {agent} on a {grid_size[0]}x{grid_size[1]} grid.

                Current positions:
                - You (Agent {agent}): ({ax},{ay})
                - Teammate (Agent {teammate}): ({bx},{by})
                - Stag: ({sx},{sy})
                - Plants: {plants_str}

                The Stag is moving toward the nearest agent. Your teammate is also trying to catch the Stag cooperatively.

                What is your next move? Think about where the Stag will be next turn, and whether your teammate can also reach it.
                ACTION:"""