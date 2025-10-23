import time
import random
from .client import client
from config import MODEL_ID, TEMPERATURE, MAX_TOKENS, SEED
from src.utils.text_utils import is_numeric_string
from src.utils.eval_utils import extract_final

# Set the random seed for reproducibility
random.seed(SEED)

# safe import of SDK exception classes
try:
    from groq._exceptions import BadRequestError, RateLimitError, APIStatusError
except Exception:
    class BadRequestError(Exception): ...
    class RateLimitError(Exception): ...
    class APIStatusError(Exception): ...

SYS_PROMPT = (
    "You are a careful math solver. Show concise reasoning in plain text.\n"
    "End with exactly one line in the SAME line format:\n"
    "Final answer: <number>\n"
    "Rules for the final line: no LaTeX, no boxes, no bold/markdown, no extra words."
)
STRICTER_ADDON = "Ensure the last line is exactly: Final answer: <number> (no TeX, no words, same line)."

def solve_math(question: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
    )
    return resp.choices[0].message.content.strip()

def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    triggers = [
        "maximum context length","max context length","context_length_exceeded",
        "context length exceeded","token limit","max_tokens","too many tokens",
        "prompt is too long","content length","context window","request too large",
    ]
    return any(t in msg for t in triggers)

def solve_math_with_retries(
    question: str,
    *,
    max_attempts: int = 5,
    wait_seconds_on_token_exceed: int = 10,
    format_retries: int = 2,
):
    """
    returns (raw_model_text, log_dict)
    log_dict: tries, format_retries_used, total_tokens, final_answer
    """
    total_tokens = 0
    last_err = None

    def _call(sys_prompt: str):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question.strip()},
                ],
            )
            usage = getattr(resp, "usage", None)
            if usage:
                tt = getattr(usage, "total_tokens", None)
                if tt is None and isinstance(usage, dict):
                    tt = usage.get("total_tokens", 0)
            else:
                tt = 0
            return resp.choices[0].message.content.strip(), int(tt)
        except Exception as e:
            raise e

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            raw, used = _call(SYS_PROMPT)
            total_tokens += used
            final = extract_final(raw)

            fr_used = 0
            while (not is_numeric_string(final)) and fr_used < format_retries:
                print("Could not finalize answer. Attempting once again")
                fr_used += 1
                raw, used = _call(SYS_PROMPT + "\n" + STRICTER_ADDON)
                total_tokens += used
                final = extract_final(raw)

            return raw, {
                "tries": attempts,
                "format_retries_used": fr_used,
                "total_tokens": total_tokens,
                "final_answer": final,
            }

        except (BadRequestError, RateLimitError, APIStatusError, Exception) as e:
            last_err = e
            if _is_token_limit_error(e) and attempts < max_attempts:
                print(f"[try {attempts}] Token limit exceeded; waiting {wait_seconds_on_token_exceed}s and retrying...")
                time.sleep(wait_seconds_on_token_exceed)
                continue
            msg = str(e).lower()
            transient = isinstance(e, (RateLimitError, APIStatusError)) or any(
                s in msg for s in ["timeout", "temporarily unavailable", "overloaded", "server error"]
            )
            print(f"[try {attempts}] error: {e.__class__.__name__}: {e}")
            if transient and attempts < max_attempts:
                time.sleep(2)
                continue
            break

    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_err}")
