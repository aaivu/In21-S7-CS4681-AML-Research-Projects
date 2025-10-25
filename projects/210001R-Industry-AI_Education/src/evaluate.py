import torch
from torch.utils.data import DataLoader

def compute_accuracy(model, dataset, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].cuda()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total
