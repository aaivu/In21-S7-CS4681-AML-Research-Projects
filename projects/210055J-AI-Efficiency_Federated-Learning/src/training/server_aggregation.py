import torch


def server_aggregation(client_weights, client_sizes):
    """Aggregate client weights using their dataset sizes.

    Parameters:
        client_weights (list[dict]): Model weights from each client.
        client_sizes (list[int]): Sizes of the datasets used by each client.

    Returns:
        dict: Aggregated global model weights.
    """
    global_model_weights = {}
    total_size = sum(client_sizes)

    # Initialize global model weights to zero
    for key in client_weights[0].keys():
        global_model_weights[key] = torch.zeros_like(client_weights[0][key])

    # Perform weighted averaging of client weights based on dataset size
    for client_weight, size in zip(client_weights, client_sizes):
        weight = size / total_size
        for key in client_weight.keys():
            global_model_weights[key] += client_weight[key] * weight

    return global_model_weights
