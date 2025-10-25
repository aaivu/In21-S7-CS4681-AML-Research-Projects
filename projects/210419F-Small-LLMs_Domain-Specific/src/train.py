from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from util import build_dataset
import json

# 1. Build dataset
dataset, label_list, label2id, id2label, tokenizer, data_collator = build_dataset(
    "../data/normalize/MACCROBAT_biomedical_ner/MACCROBAT2020-V2.json",
    "../artifacts/distilled_model"
)
train_data = dataset['train'].to_dict()  # this returns a dict of lists

# If you want a list of records instead of dict of lists
train_records = [
    {key: train_data[key][i] for key in train_data.keys()}
    for i in range(len(dataset['train']))
]

# Save to JSON
with open("train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(train_records, f, ensure_ascii=False, indent=4)

model = DistilBertForTokenClassification.from_pretrained(
    "../artifacts/distilled_model",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    eval_strategy="steps",
    save_steps=100,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 6. Save the final model
print("Saving model...")
trainer.save_model("./maccrobat_ner_model")
tokenizer.save_pretrained("./maccrobat_ner_model")