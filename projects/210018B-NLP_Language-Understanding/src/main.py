from lambada_eval import evaluate

# ---------------------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(
        n_samples=500,              # number of test samples
        k=15,                        # few-shot examples
        model="gpt-3.5-turbo-instruct",  # API model
        mode="cloze",               # cloze format is best for LAMBADA
        use_semantic=True,          # semantic example selection
        log_path="lambada_results.csv"
    )