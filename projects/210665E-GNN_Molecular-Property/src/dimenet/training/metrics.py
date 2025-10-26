import numpy as np
import tensorflow as tf


class Metrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex

        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_mae_metric = tf.keras.metrics.Mean()
        # Track per-target MAE via manual accumulators for broad TF/Keras compatibility
        self._sum_mae = tf.Variable(np.zeros(len(targets), dtype=np.float32), trainable=False)
        self._count = tf.Variable(0.0, trainable=False)

    def update_state(self, loss, mean_mae, mae, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_mae_metric.update_state(mean_mae, sample_weight=nsamples)
        # Accumulate per-target mae weighted by number of samples
        ns = tf.cast(nsamples, tf.float32)
        # mae is shape (num_targets,), accumulate sum over batch samples
        self._sum_mae.assign_add(mae * ns)
        self._count.assign_add(ns)

    def write(self):
        """Write metrics to tf.summary and the Sacred experiment."""
        for key, val in self.result().items():
            tf.summary.scalar(key, val)
            if self.ex is not None:
                if key not in self.ex.current_run.info:
                    self.ex.current_run.info[key] = []
                self.ex.current_run.info[key].append(val)

        if self.ex is not None:
            if f'step_{self.tag}' not in self.ex.current_run.info:
                self.ex.current_run.info[f'step_{self.tag}'] = []
            self.ex.current_run.info[f'step_{self.tag}'].append(tf.summary.experimental.get_step())

    def reset_states(self):
        self.loss_metric.reset_state()
        self.mean_mae_metric.reset_state()
        self._sum_mae.assign(tf.zeros_like(self._sum_mae))
        self._count.assign(0.0)

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_mae_{self.tag}', f'mean_log_mae_{self.tag}']
        keys.extend([key + '_' + self.tag for key in self.targets])
        return keys

    def result(self):
        result_dict = {}
        result_dict[f'loss_{self.tag}'] = self.loss
        result_dict[f'mean_mae_{self.tag}'] = self.mean_mae
        result_dict[f'mean_log_mae_{self.tag}'] = self.mean_log_mae
        for i, key in enumerate(self.targets):
            result_dict[key + '_' + self.tag] = self.maes[i].item()
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result().numpy().item()

    @property
    def maes(self):
        # Avoid division by zero
        count = self._count.numpy().item()
        if count == 0.0:
            return np.zeros(len(self.targets), dtype=np.float32)
        return (self._sum_mae.numpy() / count)

    @property
    def mean_mae(self):
        return self.mean_mae_metric.result().numpy().item()

    @property
    def mean_log_mae(self):
        return np.mean(np.log(self.maes)).item()
