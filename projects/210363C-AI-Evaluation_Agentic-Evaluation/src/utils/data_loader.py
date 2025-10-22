import pandas as pd

def load_dataset(path: str):
    df = pd.read_json(path, lines=True)
    need_cols = ["id", "problem", "answer"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in JSONL: {missing}")
    df.rename(columns={"problem": "question", "answer": "gold_answer"}, inplace=True)
    return df
