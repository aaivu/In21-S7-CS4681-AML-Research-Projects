import time
import torch
from torch.utils.data import DataLoader
from src.evaluate import compute_accuracy

def train_model(model, train_dataset, test_dataset, epochs=3, lr=5e-5, batch_size=16):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs["loss"].backward()
            optimizer.step()

        acc = compute_accuracy(model, test_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc:.4f}")

    total_time = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    final_acc = compute_accuracy(model, test_dataset)
    return {"accuracy": final_acc, "train_time_sec": round(total_time,2), "max_memory_MB": round(max_memory,2)}
