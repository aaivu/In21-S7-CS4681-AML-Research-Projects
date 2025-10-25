import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# -------------------------------
# 1. Load MACCROBAT dataset
# -------------------------------
def load_maccrobat(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]

# -------------------------------
# 2. Build label map (BIO scheme)
# -------------------------------
def build_label_map(data):
    unique_labels = set()
    for example in data:
        for ent in example["ner_info"]:
            unique_labels.add(f"B-{ent['label']}")
            unique_labels.add(f"I-{ent['label']}")
    unique_labels.add("O")

    label_list = sorted(list(unique_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id, id2label

# -------------------------------
# 3. Encode example (BIO labels)
# -------------------------------
def encode_example(example, tokenizer, label2id, max_length=512):
    text = example["full_text"]
    entities = example["ner_info"]

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False   # <-- let collator handle dynamic padding
    )

    labels = ["O"] * len(encoding["input_ids"])

    for ent in entities:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        for idx, (start, end) in enumerate(encoding["offset_mapping"]):
            if start == 0 and end == 0:  # special tokens
                continue
            if start >= ent_start and end <= ent_end:
                if start == ent_start:
                    labels[idx] = f"B-{ent_label}"
                else:
                    labels[idx] = f"I-{ent_label}"

    # convert to ids
    label_ids = [label2id.get(l, label2id["O"]) for l in labels]
    encoding["labels"] = label_ids

    # remove offset mapping (not needed for training)
    encoding.pop("offset_mapping")
    return encoding

# -------------------------------
# 4. Build Hugging Face Dataset
# -------------------------------
def build_dataset(json_path, tokenizer_path, train_split=0.8):
    # load raw data
    data = load_maccrobat(json_path)

    # build label map
    label_list, label2id, id2label = build_label_map(data)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # encode all examples
    encoded_data = [encode_example(ex, tokenizer, label2id) for ex in data]

    # convert to HuggingFace Dataset
    dataset = Dataset.from_list(encoded_data)

    # train/validation split
    dataset = dataset.train_test_split(test_size=1-train_split)

    # data collator for token classification (handles padding of both inputs + labels)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return dataset, label_list, label2id, id2label, tokenizer, data_collator
