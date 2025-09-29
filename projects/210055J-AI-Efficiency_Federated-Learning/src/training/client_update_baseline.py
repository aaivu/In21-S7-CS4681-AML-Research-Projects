import copy
import torch.optim as optim
from utils.loss_function import cross_entropy_loss


def client_update_baseline(global_model, local_data, learning_rate, device, local_epochs):
    """Perform local training for a single client using only cross-entropy loss.

    Parameters:
        global_model: The global model to be cloned for local training.
        local_data: DataLoader containing the client's local data.
        learning_rate (float): Learning rate for the optimizer.
        device: Device on which to perform training.
        local_epochs (int): Number of local epochs to train the model.

    Returns:
        dict: Updated state dictionary of the locally trained model.
    """
    # Clone the global model so the original weights remain unchanged
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

    for _ in range(local_epochs):
        for inputs, labels in local_data:
            inputs, labels = inputs.to(device), labels.to(device)
            # If labels come as sequences, focus on the next character only
            if labels.dim() > 1:
                labels = labels[:, -1]
            outputs = local_model(inputs)
            loss = cross_entropy_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return local_model.state_dict()
