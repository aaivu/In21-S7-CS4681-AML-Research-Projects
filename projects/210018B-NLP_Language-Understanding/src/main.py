from lambada_eval import evaluate

# ---------------------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(
        n_samples=1000,             # number of test samples (max=5153)
        k=1,                        # few-shot examples
        model="gpt-3.5-turbo-instruct",  # API model
        mode="cloze",               # "cloze", "default"
        use_POS_sem=True,         # POS and semantic example selection
        alpha=0.2,                   # weight of semantic component
        log_path="lambada_results.csv"
    )