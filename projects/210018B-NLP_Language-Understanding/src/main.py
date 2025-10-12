from lambada_eval import evaluate

# ---------------------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(
        n_samples=1000,              # number of test samples
        k=1,                        # few-shot examples
        model="gpt-3.5-turbo-instruct",  # API model
        mode="cloze",               # "cloze", "default"
        use_semantic=True,          # semantic example selection
        log_path="lambada_results.csv"
    )