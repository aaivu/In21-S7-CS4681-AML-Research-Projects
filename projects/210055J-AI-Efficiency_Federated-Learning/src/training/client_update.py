import torch.optim as optim
import torch
from utils.loss_function import distillation_loss, cross_entropy_loss
from models.student_model import StudentModel
from models.teacher_model import TeacherModel


def client_update(global_model, local_data, lambda_, T, tau, learning_rate, device, local_epochs):
    """Perform local training for a single client using a fresh teacher and student.

    Parameters:
        global_model: The global model to be used as a starting point for the student and teacher models.
        local_data: The data loader containing the client's local data.
        lambda_ (float): Weight for the distillation loss component.
        T (float): Temperature parameter for knowledge distillation.
        tau (float): Confidence threshold for applying distillation.
        learning_rate (float): Learning rate for the student's optimizer.
        device: Device on which training is performed.
        local_epochs (int): Number of local epochs to train the student.
    """
    teacher = TeacherModel(global_model).to(device)
    student = StudentModel(global_model).to(device)
    teacher.freeze()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=learning_rate)

    for _ in range(local_epochs):
        for inputs, labels in local_data:
            inputs, labels = inputs.to(device), labels.to(device)
            # If labels are provided as sequences, use only the next token
            if labels.dim() > 1:
                labels = labels[:, -1]
            # Teacher and student predictions (raw logits)
            teacher_out = teacher(inputs)
            student_out = student(inputs)

            # Calculate cross-entropy on raw student logits
            ce_loss = cross_entropy_loss(student_out, labels)

            # Compute teacher probabilities and confidence
            teacher_probs = torch.softmax(teacher_out / T, dim=1)
            conf, _ = teacher_probs.max(dim=1)

            # Apply distillation only when average confidence exceeds the threshold
            if conf.mean() >= tau:
                kd_loss = distillation_loss(teacher_out, student_out, T)
                loss = ce_loss + lambda_ * kd_loss
            else:
                loss = ce_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Return only the underlying model's parameters to avoid the 'model.' prefix
    return student.model.state_dict()
