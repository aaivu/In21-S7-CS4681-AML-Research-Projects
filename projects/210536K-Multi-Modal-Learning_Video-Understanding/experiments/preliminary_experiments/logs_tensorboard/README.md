# Training Logs

This folder contains the training logs recorded during each experiment for the Multi-Modal Learning Video Understanding project. These logs capture important training metrics and parameters that were monitored throughout the training process.

## Contents

The logs in this folder include:

- Training loss curves
- Validation metrics
- Learning rate schedules
- Other parameter behaviors during training
- Performance metrics over epochs

## Viewing with TensorBoard

To visualize the training progress and analyze the recorded metrics, you can use TensorBoard:

```bash
tensorboard --logdir=./
```

This will launch TensorBoard in your browser where you can:

- View training and validation loss curves
- Monitor learning rate changes
- Analyze other training parameters
- Compare different experimental runs
- Track model performance over time

## Usage

1. Navigate to this logs directory
2. Run the TensorBoard command above
3. Open the provided URL in your browser
4. Explore the various tabs to analyze training behavior

These logs are essential for understanding the training dynamics and can help in reproducing the experimental results or debugging training issues.
