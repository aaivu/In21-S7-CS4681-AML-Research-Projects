import re

def extract_final_answer(model_output: str) -> str:
    """
    Extracts the final answer from the model output string.
    Tries to find the last integer/float in the text.
    If nothing is found, returns an empty string.
    """
    if not model_output:
        return ""

    # Look for numbers (int or float) in the text
    matches = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if matches:
        return matches[-1]  # last number is often the final answer

    # Fallback: return stripped text
    return model_output.strip()
