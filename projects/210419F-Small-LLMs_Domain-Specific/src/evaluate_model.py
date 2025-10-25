import numpy as np
import json
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from util import build_dataset
import evaluate

# Load seqeval for NER metrics
seqeval = evaluate.load("seqeval")

# 1. Reload dataset (tokenized, already has input_ids, attention_mask, labels)
dataset, label_list, label2id, id2label, tokenizer, data_collator = build_dataset(
    "../data/normalize/MACCROBAT_biomedical_ner/MACCROBAT2020-V2.json",
    "./maccrobat_ner_model"   # use trained tokenizer
)

model = DistilBertForTokenClassification.from_pretrained("./maccrobat_ner_model", num_labels=len(label_list))


# dataset, label_list, label2id, id2label, tokenizer, data_collator = build_dataset(
#     "../data/normalize/MACCROBAT_biomedical_ner/MACCROBAT2020-V2.json",
#     "distilbert-base-uncased"   # use trained tokenizer
# )

# # 2. Reload trained model


# 3. Metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 4. Trainer (evaluation only)
training_args = TrainingArguments(output_dir="./results")

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. Run prediction on test set
predictions, labels, _ = trainer.predict(dataset["test"])
predictions = np.argmax(predictions, axis=2)

# 6. Convert input_ids to tokens
all_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in dataset["test"]["input_ids"]]

# 7. Align predictions and true labels
true_labels = [
    [id2label[l] for l in label if l != -100]
    for label in labels
]
true_predictions = [
    [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# 8. Save results as JSON
all_results = []
for tokens, preds, golds in zip(all_tokens, true_predictions, true_labels):
    sentence_result = []
    for tok, pred, gold in zip(tokens, preds, golds):
        # Skip special tokens
        if tok in tokenizer.all_special_tokens:
            continue
        sentence_result.append({
            "token": tok,
            "prediction": pred,
            "label": gold
        })
    all_results.append(sentence_result)

with open("ner_predictions.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

# 9. Print metrics
results = trainer.evaluate()
print("Evaluation results:", results)
print("Predictions saved to ner_predictions.json")
