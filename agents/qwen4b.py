from ollama import chat
import re, random
from utils.const import ACTION_MAP


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