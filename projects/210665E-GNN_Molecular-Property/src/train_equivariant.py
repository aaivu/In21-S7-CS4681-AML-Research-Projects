"""
Training script: quick baseline vs E(3)-equivariant DimeNet++ benchmark.

Goal: show improvement with minimal time/hardware by training on a tiny subset
for a few steps using the existing Trainer/DataProvider utilities.
"""

import argparse
import os
import json
import time
import numpy as np
import tensorflow as tf

from dimenet.model.dimenet_pp import DimeNetPP as DimeNetBaseline
from dimenet.model.dimenet_pp_equivariant import DimeNetPPEquivariant
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics


class EnergyOnlyWrapper(tf.keras.Model):
	"""Wrap a model that returns a dict to expose only energy tensor output."""
	def __init__(self, eq_model: tf.keras.Model):
		super().__init__(name="energy_only_wrapper")
		self.eq_model = eq_model

	def call(self, inputs, training=False):
		outputs = self.eq_model(inputs, training=training)
		# Equivariant model returns {'energy': (n_atoms or batch, T), 'forces': optional}
		# Trainer expects shape (batch, T). Inputs include 'batch_seg'.
		energy = outputs['energy']
		return energy


def make_provider(data_path: str, cutoff: float, targets, ntrain: int, nval: int, batch_size: int) -> DataProvider:
	container = DataContainer(data_path, cutoff=cutoff, target_keys=targets)
	provider = DataProvider(container, ntrain=ntrain, nvalid=nval, batch_size=batch_size, seed=42, randomized=True)
	return provider


def quick_train(model: tf.keras.Model, provider: DataProvider, steps_per_epoch: int, val_steps: int, epochs: int, learning_rate: float, targets, patience: int = 20, save_best_path: str = None) -> dict:
	metrics_train = Metrics('train', targets=provider.data_container.target_keys)
	metrics_val = Metrics('val', targets=provider.data_container.target_keys)

	trainer = Trainer(model, learning_rate=learning_rate, warmup_steps=3000, decay_steps=4000000, decay_rate=0.01)

	train_iter = iter(provider.get_dataset('train'))
	val_iter = iter(provider.get_dataset('val'))

	train_mae_history = []
	val_mae_history = []
	best_val = float('inf')
	no_improve = 0
	best_saved_path = None

	for epoch in range(epochs):
		metrics_train.reset_states()
		for _ in range(steps_per_epoch):
			trainer.train_on_batch(train_iter, metrics_train)
		trainer.load_averaged_variables()
		metrics_val.reset_states()
		for _ in range(val_steps):
			trainer.test_on_batch(val_iter, metrics_val)
		trainer.restore_variable_backups()

		print(f"Epoch {epoch+1}/{epochs} - train MAE: {metrics_train.mean_mae:.6f} - val MAE: {metrics_val.mean_mae:.6f}")

		train_mae_history.append(metrics_train.mean_mae)
		val_mae_history.append(metrics_val.mean_mae)

		# Early stopping on validation MAE
		if metrics_val.mean_mae + 1e-12 < best_val:
			best_val = metrics_val.mean_mae
			no_improve = 0
			# Save best weights if a path is provided
			if save_best_path:
				model.save_weights(save_best_path)
				best_saved_path = save_best_path
				print(f"Saved best model to: {save_best_path}")
		else:
			no_improve += 1
			if no_improve >= patience:
				print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs). Best val MAE: {best_val:.6f}")
				break

	return {
		'mean_mae_train': metrics_train.mean_mae,
		'mean_mae_val': metrics_val.mean_mae,
		'maes_val': metrics_val.maes,
		'train_history': train_mae_history,
		'val_history': val_mae_history,
		'best_val_mae': best_val,
		'best_model_path': best_saved_path,
	}


def evaluate_on_split(model: tf.keras.Model, provider: DataProvider, split: str) -> dict:
	"""Evaluate model on a dataset split and return MAE metrics."""
	metrics = Metrics(split, targets=provider.data_container.target_keys)
	n_samples = provider.nsamples[split]
	batch_size = provider.batch_size
	steps = max(1, int(np.ceil(n_samples / batch_size)))
	trainer = Trainer(model, learning_rate=1e-3)
	iterator = iter(provider.get_dataset(split))
	metrics.reset_states()
	for _ in range(steps):
		trainer.test_on_batch(iterator, metrics)
	return {
		'mean_mae': metrics.mean_mae,
		'maes': metrics.maes,
	}


# def build_baseline(config: dict) -> tf.keras.Model:
# 	return DimeNetBaseline(
# 		emb_size=config['emb_size'],
# 		out_emb_size=config['out_emb_size'],
# 		int_emb_size=config['int_emb_size'],
# 		basis_emb_size=config['basis_emb_size'],
# 		num_blocks=config['num_blocks'],
# 		num_spherical=config['num_spherical'],
# 		num_radial=config['num_radial'],
# 		cutoff=config['cutoff'],
# 		envelope_exponent=config['envelope_exponent'],
# 		num_before_skip=config['num_before_skip'],
# 		num_after_skip=config['num_after_skip'],
# 		num_dense_output=config['num_dense_output'],
# 		num_targets=config['num_targets'],
# 		activation=tf.nn.swish,
# 		extensive=True,
# 		output_init=config.get('output_init', 'zeros'),
# 	)


def build_equivariant(config: dict) -> tf.keras.Model:
	eq_core = DimeNetPPEquivariant(
		emb_size=config['emb_size'],
		out_emb_size=config['out_emb_size'],
		int_emb_size=config['int_emb_size'],
		basis_emb_size=config['basis_emb_size'],
		num_blocks=config['num_blocks'],
		num_spherical=config['num_spherical'],
		num_radial=config['num_radial'],
		cutoff=config['cutoff'],
		envelope_exponent=config['envelope_exponent'],
		num_before_skip=config['num_before_skip'],
		num_after_skip=config['num_after_skip'],
		num_dense_output=config['num_dense_output'],
		num_targets=config['num_targets'],
		l_max=config.get('l_max', 1),
		activation=tf.nn.swish,
		extensive=True,
		output_init=config.get('output_init', 'zeros'),
		predict_forces=False,
	)
	# Wrap to return energy for Trainer compatibility
	return EnergyOnlyWrapper(eq_core)


def main():
	parser = argparse.ArgumentParser(description='Quick benchmark: baseline vs equivariant DimeNet++')
	parser.add_argument('--data_path', type=str, default='data/qm9_eV.npz')
	parser.add_argument('--targets', nargs='+', default=['U0'])
	parser.add_argument('--epochs', type=int, default=300, help='Number of epochs (paper uses 300)')
	parser.add_argument('--ntrain', type=int, default=110000, help='Number of training samples (paper uses 110000)')
	parser.add_argument('--nval', type=int, default=10000, help='Number of validation samples (paper uses 10000)')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size (paper uses 32)')
	parser.add_argument('--steps_per_epoch', type=int, default=10000, help='Steps per epoch (paper uses ~10000)')
	parser.add_argument('--val_steps', type=int, default=1000, help='Validation steps per epoch')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (paper uses 0.001)')
	parser.add_argument('--cutoff', type=float, default=5.0)
	parser.add_argument('--output_dir', type=str, default='./outputs')
	parser.add_argument('--quick_test', action='store_true', help='Use small dataset for quick testing (overrides ntrain/nval/epochs)')
	args = parser.parse_args()

	# Quick test mode for development
	if args.quick_test:
		print("Quick test mode: using small dataset and few epochs")
		args.ntrain = 1000
		args.nval = 300
		args.epochs = 2
		args.steps_per_epoch = 30
		args.val_steps = 10
		args.batch_size = 16

	# Reproducibility
	tf.random.set_seed(42)
	np.random.seed(42)
	os.makedirs(args.output_dir, exist_ok=True)

	# Optimize TensorFlow for CPU performance
	tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
	tf.config.threading.set_intra_op_parallelism_threads(0)   # Use all CPU cores
	
	# Check for GPU availability
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		print(f"GPU detected: {len(gpus)} device(s)")
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			print("GPU memory growth enabled")
		except RuntimeError as e:
			print(f"GPU setup error: {e}")
	else:
		print("No GPU detected - using CPU (will be slower)")
		print("Consider using Google Colab for GPU acceleration")

	print(f"Training configuration:")
	print(f"  Epochs: {args.epochs}")
	print(f"  Training samples: {args.ntrain}")
	print(f"  Validation samples: {args.nval}")
	print(f"  Batch size: {args.batch_size}")
	print(f"  Steps per epoch: {args.steps_per_epoch}")
	print(f"  Total training steps: {args.epochs * args.steps_per_epoch}")
	print()

	# Set both models to DimeNet++ paper's commonly used hyperparameters
	# emb_size=128, out_emb_size=256, int_emb_size=64, basis_emb_size=8,
	# num_blocks=4, num_spherical=7, num_radial=6, envelope_exponent=5, cutoff=5.0,
	# num_before_skip=1, num_after_skip=2, num_dense_output=3
	config = {
        'emb_size': 128,
        'out_emb_size': 256,
        'int_emb_size': 64,
        'basis_emb_size': 8,
        'num_blocks': 4,
        'num_bilinear': 8,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': args.cutoff,
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_dense_output': 3,
        'num_targets': len(args.targets),
        'targets': args.targets,
        'output_init': 'zeros',
        'l_max': 1,
    }

	provider = make_provider(args.data_path, args.cutoff, args.targets, args.ntrain, args.nval, args.batch_size)

	
	# Equivariant: use same hyperparameters to isolate effect of equivariance
	equiv_config = dict(config)
	equivariant = build_equivariant(equiv_config)

	# Use same schedule and LR for fair comparison
	epochs_eq = args.epochs
	steps_eq = args.steps_per_epoch
	val_steps_eq = args.val_steps
	lr_eq = args.learning_rate
	best_equiv_path = os.path.join(args.output_dir, 'equivariant_best.weights.h5')
	res_equiv = quick_train(equivariant, provider, steps_eq, val_steps_eq, epochs_eq, lr_eq, args.targets, patience=20, save_best_path=best_equiv_path)

    # Baseline
	# baseline = build_baseline(config)
	# best_base_path = os.path.join(args.output_dir, 'baseline_best.weights.h5')
	# res_base = quick_train(baseline, provider, args.steps_per_epoch, args.val_steps, args.epochs, args.learning_rate, args.targets, patience=10, save_best_path=best_base_path)

	# Load best weights and evaluate on test split
	# if res_base.get('best_model_path'):
	# 	baseline.load_weights(res_base['best_model_path'])
	if res_equiv.get('best_model_path'):
		equivariant.load_weights(res_equiv['best_model_path'])

	# test_baseline = evaluate_on_split(baseline, provider, 'test')
	test_equivariant = evaluate_on_split(equivariant, provider, 'test')

	print(' benchmark:')
	# print(f"Baseline DimeNet     - val mean MAE: {res_base['mean_mae_val']:.6f} eV ({res_base['mean_mae_val']*1000:.2f} meV)")
	print(f"Equivariant DimeNet  - val mean MAE: {res_equiv['best_val_mae']:.6f} eV")
	# print(f"Baseline DimeNet     - test mean MAE: {test_baseline['mean_mae']:.6f} eV ({test_baseline['mean_mae']*1000:.2f} meV)")
	print(f"Equivariant DimeNet  - test mean MAE: {test_equivariant['mean_mae']:.6f} eV")
	# print(f"\nPaper results (DimeNet++): 6.32 meV")
	# print(f"Your baseline: {test_baseline['mean_mae']*1000:.2f} meV (factor: {test_baseline['mean_mae']*1000/6.32:.1f}x higher)")
	# print(f"Your equivariant: {test_equivariant['mean_mae']*1000:.2f} meV (factor: {test_equivariant['mean_mae']*1000/6.32:.1f}x higher)")

	# Report relative improvement and save results
	# impr = (res_base['mean_mae_val'] - res_equiv['mean_mae_val']) / max(res_base['mean_mae_val'], 1e-8)
	# print(f"Relative improvement (lower is better): {impr * 100:.2f}%")

	results = {
		'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
		'config_baseline': {
            'emb_size': 128,
            'out_emb_size': 256,
            'int_emb_size': 64,
            'basis_emb_size': 8,
            'num_blocks': 4,
            'num_bilinear': 8,
            'num_spherical': 7,
            'num_radial': 6,
            'l_max': 0,
		},
		'config_equivariant': {
            'emb_size': 128,
            'out_emb_size': 256,
            'int_emb_size': 64,
            'basis_emb_size': 8,
            'num_blocks': 4,
            'num_spherical': 7,
            'num_radial': 6,
            'l_max': 1,
            'epochs': int(args.epochs),
            'steps_per_epoch': int(args.steps_per_epoch),
            'learning_rate': float(args.learning_rate),
		},
		'provider': {
			'data_path': args.data_path,
			'ntrain': args.ntrain,
			'nval': args.nval,
			'batch_size': args.batch_size,
			'cutoff': args.cutoff,
		},
		# 'baseline': res_base,
		'equivariant': res_equiv,
		'test_metrics': {
			# 'baseline': test_baseline,
			'equivariant': test_equivariant,
		},
		'test_metrics_mev': {
			# 'baseline': {k: v*1000 if k == 'mean_mae' else v*1000 for k, v in test_baseline.items()},
			'equivariant': {k: v*1000 if k == 'mean_mae' else v*1000 for k, v in test_equivariant.items()},
		},
	}
	def _to_json(o):
		import numpy as _np
		if isinstance(o, (_np.ndarray,)):
			return o.tolist()
		if isinstance(o, (_np.floating,)):
			return float(o)
		if isinstance(o, (_np.integer,)):
			return int(o)
		return str(o)
	with open(os.path.join(args.output_dir, 'benchmark_results.json'), 'w') as f:
		json.dump(results, f, indent=2, default=_to_json)

	# Plot error vs epoch and save
	try:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(7, 4))
		# plt.plot(res_base['train_history'], label='Baseline train MAE', linestyle='--')
		# plt.plot(res_base['val_history'], label='Baseline val MAE')
		plt.plot(res_equiv['train_history'], label='Equivariant train MAE', linestyle='--')
		plt.plot(res_equiv['val_history'], label='Equivariant val MAE')
		plt.xlabel('Epoch')
		plt.ylabel('MAE')
		plt.title('Error vs Epoch (Equivariant)')
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(args.output_dir, 'error_vs_epoch.png'), dpi=150)
		plt.close()
	except Exception as e:
		print(f"Plotting failed: {e}")


if __name__ == '__main__':
	main()
