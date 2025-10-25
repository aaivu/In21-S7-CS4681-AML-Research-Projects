"""
Dense retrieval model implementation using RoBERTa.
"""
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class DenseRetriever(nn.Module):
    def __init__(self, model_name="roberta-base", device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.query_encoder = RobertaModel.from_pretrained(model_name)
        self.doc_encoder = RobertaModel.from_pretrained(model_name)
        self.to(device)

    def encode_query(self, query, max_length=128):
        inputs = self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.query_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def encode_doc(self, doc, max_length=512):
        inputs = self.tokenizer(
            doc,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.doc_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def forward(self, query_batch, doc_batch):
        query_embeds = self.encode_query(query_batch)
        doc_embeds = self.encode_doc(doc_batch)
        return query_embeds, doc_embeds

class MDRTrainer:
    def __init__(self, model, lr=2e-5, warmup_steps=1000):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=warmup_steps
        )

    def train_step(self, query_batch, pos_docs, neg_docs, temperature=0.1):
        query_embeds, pos_doc_embeds = self.model(query_batch, pos_docs)
        _, neg_doc_embeds = self.model(query_batch, neg_docs)

        # Compute similarity scores
        pos_scores = torch.matmul(query_embeds, pos_doc_embeds.t()) / temperature
        neg_scores = torch.matmul(query_embeds, neg_doc_embeds.t()) / temperature

        # Contrastive loss
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(len(query_batch)).long().to(self.model.device)
        loss = nn.CrossEntropyLoss()(scores, labels)

        # Update
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()