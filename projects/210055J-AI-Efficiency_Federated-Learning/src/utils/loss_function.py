import torch
import torch.nn.functional as F

def cross_entropy_loss(student_out, labels):
    """
    Computes the standard cross-entropy loss between the student model's predictions and the true labels.
    
    Parameters:
        student_out (torch.Tensor): Predictions from the student model.
        labels (torch.Tensor): Ground truth labels.
    
    Returns:
        torch.Tensor: Computed cross-entropy loss.
    """
    return F.cross_entropy(student_out, labels)

def distillation_loss(teacher_output, student_output, T):
    """
    Calculate the knowledge distillation loss between teacher and student outputs.
    
    Parameters:
        teacher_output (torch.Tensor): Output from the teacher model (frozen).
        student_output (torch.Tensor): Output from the student model (trainable).
        T (float): Temperature parameter for scaling the outputs.
    
    Returns:
        torch.Tensor: Computed knowledge distillation loss.
    """
    # Softmax over temperature-scaled logits
    teacher_prob = F.softmax(teacher_output / T, dim=1)

    # KL divergence between teacher and student outputs
    loss = F.kl_div(
        F.log_softmax(student_output / T, dim=1),
        teacher_prob,
        reduction='batchmean',
    ) * (T ** 2)
    return loss

def total_loss(student_out, labels, teacher_output, T, lambda_):
    """
    Combines both cross-entropy and knowledge distillation loss for the final loss.
    
    Parameters:
        student_out (torch.Tensor): Student model's predictions.
        labels (torch.Tensor): Ground truth labels.
        teacher_output (torch.Tensor): Teacher model's predictions.
        T (float): Temperature parameter for distillation.
        lambda_ (float): Weight for the distillation loss.
    
    Returns:
        torch.Tensor: Combined total loss.
    """
    ce_loss = cross_entropy_loss(student_out, labels)
    kd_loss = distillation_loss(teacher_output, student_out, T)
    return ce_loss + lambda_ * kd_loss
