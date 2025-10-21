################################################################################
# Training and Evaluation Utilities for LSTM + MoE Language Model
################################################################################
import time
import math
import torch

################################################################################
# Parameter and Metric Utilities
################################################################################
def count_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ppl_per_million(val_ppl, n_params):
    """Compute perplexity normalized by model size."""
    return val_ppl / (n_params / 1e6)

def time_per_million(epoch_time, n_params):
    """Compute epoch training time normalized by model size."""
    return epoch_time / (n_params / 1e6)

################################################################################
# Evaluation Loop
################################################################################
def evaluate(model, data_source, criterion, seq_len, device, get_batch):
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = None
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len)
            outputs, hidden, aux_loss = model(data, hidden)

            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            batch, seqlen, vocab = outputs.size()
            loss = criterion(outputs.view(-1, vocab), targets.reshape(-1))

            total_loss += loss.item() * seqlen
            total_tokens += seqlen

    return total_loss / total_tokens  # return average loss
################################################################################
# Training Loop
################################################################################
def train_epoch(model, train_data, optimizer, criterion, seq_len, clip, device,
                get_batch, aux_coef=1.0, log_interval=200, accum_steps=4):
    """One training epoch with gradient accumulation and optional MoE loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    hidden = None
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for step, i in enumerate(range(0, train_data.size(1) - 1, seq_len)):
        data, targets = get_batch(train_data, i, seq_len)

        with torch.cuda.amp.autocast():
            outputs, hidden, aux_loss = model(data, hidden)
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            batch_sz, seqlen, vocab = outputs.size()
            loss = criterion(outputs.view(-1, vocab), targets.reshape(-1))
            total = (loss + aux_coef * aux_loss) / accum_steps

        scaler.scale(total).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * seqlen
        total_tokens += seqlen

        if step % log_interval == 0 and step > 0:
            cur_loss = total_loss / total_tokens
            elapsed = time.time() - start_time
            print(f'| Batch {step:5d} | lr {optimizer.param_groups[0]["lr"]:.2e} '
                  f'| ms/batch {elapsed * 1000.0 / log_interval:.2f} '
                  f'| loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}')
            start_time = time.time()

    avg_loss = total_loss / total_tokens
    return avg_loss
