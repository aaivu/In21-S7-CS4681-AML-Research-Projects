from src.utils.text_utils import normalize_scalar, to_fraction, FINAL_RE

def answers_match(pred: str, gold: str) -> bool:
    p = normalize_scalar(pred); g = normalize_scalar(gold)
    if p == g: return True
    pf, gf = to_fraction(p), to_fraction(g)
    return pf is not None and gf is not None and pf == gf

def extract_final(text: str) -> str:
    if not text: return ""
    last = None
    for m in FINAL_RE.finditer(text):
        last = m
    if last:
        tail = last.group(1).strip()
        if tail: return normalize_scalar(tail)
        for ln in text[last.end():].splitlines():
            ln = ln.strip()
            if ln: return normalize_scalar(ln)
    return ""
