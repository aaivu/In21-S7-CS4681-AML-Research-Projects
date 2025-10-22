import re
from fractions import Fraction

FINAL_RE = re.compile(r"^\s*Final\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def normalize_scalar(x: str) -> str:
    if x is None: return ""
    s = str(x).replace("−","-").replace("—","-").replace("–","-").replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    s = re.sub(r"\$(.*?)\$", r"\1", s)
    s = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", s)
    return s

def is_numeric_string(s: str) -> bool:
    if not s: return False
    s = normalize_scalar(s)
    pats = [
        r"[+-]?\d+",
        r"[+-]?\d*\.\d+",
        r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+",
        r"[+-]?\d+\s*/\s*[+-]?\d+",
        r"[+-]?\d+\s+\d+\s*/\s*\d+",
    ]
    return any(re.fullmatch(p, s) for p in pats)

def to_fraction(s: str):
    s = normalize_scalar(s)
    m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", s)
    if m:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den == 0: return None
        return Fraction(whole,1) + (Fraction(num, den) if whole>=0 else -Fraction(num, den))
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", s):
        num, den = re.split(r"/", s.replace(" ", ""))
        den = int(den)
        if den == 0: return None
        return Fraction(int(num), den)
    try:
        return Fraction(s)
    except Exception:
        return None
