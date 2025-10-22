from datasets import load_dataset
from transformers import AutoTokenizer

def load_imdb_subset(train_size=5000, test_size=5000, max_length=128):
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(test_size))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_dataset, test_dataset
