import pandas as pd
from src.model.solver import solve_math_with_retries
from src.utils.parser import extract_final_answer

def evaluate_dataset(df):
    records = []
    for _, row in df.iterrows():
        qid, q, gold = row["id"], row["question"], row["gold_answer"]

        raw_out, log = solve_math_with_retries(q)
        final_ans = log.get("final_answer", None)

        correct = (str(final_ans).strip() == str(gold).strip())

        # Calculate efficiency metrics
        total_tokens = log.get("total_tokens", 0)
        llm_calls = log.get("tries", 1)  # Updated to use `tries` from log
        walltime_sec = log.get("walltime_sec", 0)
        exec_successes = log.get("exec_successes", 0)
        solved = 1 if correct else 0

        tps = llm_calls / solved if solved else 0
        eps = exec_successes / solved if solved else 0
        te = solved / (total_tokens / 1000) if total_tokens > 0 else 0
        ce = solved / llm_calls if llm_calls > 0 else 0
        le = solved / (walltime_sec / 60) if walltime_sec > 0 else 0

        records.append({
            "id": qid,
            "question": q,
            "gold_answer": gold,
            "model_output": raw_out,
            "final_answer": final_ans,
            "correct": correct,
            "api_tries": log.get("tries", 0),
            "format_retries_used": log.get("format_retries_used", 0),
            "llm_calls": llm_calls,
            "total_tokens": total_tokens,
            "walltime_sec": walltime_sec,
            "exec_successes": exec_successes,
            "TPS": tps,
            "EPS": eps,
            "TE": te,
            "CE": ce,
            "LE": le,
        })

    results_df = pd.DataFrame(records)

    # Print accuracy summary
    accuracy = results_df["correct"].mean()
    print(f"Accuracy: {accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})")

    return results_df
