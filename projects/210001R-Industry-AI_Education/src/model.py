import torch
import torch.nn as nn
from transformers import AutoModel

class HybridAttentionClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, use_flash=True, use_linear=True):
        super().__init__()
        self.use_flash = use_flash
        self.use_linear = use_linear

        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        if self.use_linear:
            print(" Linear attention enabled (conceptual)")
        if self.use_flash:
            print(" FlashAttention enabled (conceptual)")

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:,0,:]  # CLS token
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"logits": logits, "loss": loss}
