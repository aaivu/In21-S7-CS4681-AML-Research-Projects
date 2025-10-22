"""
Equivariant output block for DimeNet++.

This module implements the equivariant version of the output block that produces
rotationally invariant energies and equivariant forces.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .equivariant_utils import equivariant_aggregation
from ..initializers import GlorotOrthogonal


class EquivariantOutputPPBlock(layers.Layer):
    """
    Equivariant output block that produces invariant energies and equivariant forces.
    
    This block ensures that:
    - Energy predictions are rotationally invariant (scalar)
    - Force predictions are rotationally equivariant (vector)
    """
    
    def __init__(self, emb_size, out_emb_size, num_dense_output, num_targets, 
                 l_max=1, activation=None, output_init='zeros', name='equivariant_output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.out_emb_size = out_emb_size
        self.num_dense_output = num_dense_output
        self.num_targets = num_targets
        self.l_max = l_max
        self.weight_init = GlorotOrthogonal()

        # RBF projection to match embedding size (like OutputPPBlock)
        self.dense_rbf = layers.Dense(emb_size, use_bias=False,
                                      kernel_initializer=self.weight_init)

        # Dense layers for scalar (energy) prediction on aggregated atom features
        self.dense_layers = []
        for i in range(num_dense_output):
            self.dense_layers.append(
                layers.Dense(out_emb_size, activation=activation, use_bias=True,
                            kernel_initializer=self.weight_init))
        
        # Final output layer for energies (scalar, invariant)
        self.output_layer = layers.Dense(num_targets, activation=None, use_bias=True,
                                        kernel_initializer=output_init)
        
        # Equivariant transformations for force prediction
        if l_max >= 1:
            # Simple linear map for vector-to-vector transformation
            self.force_transforms = {}
            for l in range(1, l_max + 1):
                self.force_transforms[l] = layers.Dense(3, use_bias=False, kernel_initializer=self.weight_init)

    def call(self, inputs):
        """
        Forward pass of the equivariant output block.
        
        Parameters
        ----------
        inputs : list
            [features, rbf, idnb_i, n_atoms, R] where:
            - features: dictionary of irreducible tensor features
            - rbf: radial basis functions
            - idnb_i: neighbor indices
            - n_atoms: number of atoms
            - R: atomic positions (for force computation)
            
        Returns
        -------
        dict
            Dictionary containing 'energy' and optionally 'forces'
        """
        features, rbf, idnb_i, n_atoms, R = inputs
        
        # Process scalar features for energy prediction
        if 0 in features:
            x_scalar = features[0]  # Shape: (n_bonds, emb_size)

            # Project and apply radial basis to match emb_size then modulate bonds
            g = self.dense_rbf(rbf)  # Shape: (n_bonds, emb_size)
            x_scalar = g * x_scalar

            # Aggregate bond contributions to atoms
            x_atoms = tf.math.unsorted_segment_sum(x_scalar, idnb_i, n_atoms)

            # Atom-wise MLP to out_emb_size then final projection to targets
            for layer in self.dense_layers:
                x_atoms = layer(x_atoms)

            energy = self.output_layer(x_atoms)  # Shape: (n_atoms, num_targets)
        else:
            # If no scalar features, return zeros
            energy = tf.zeros((n_atoms, self.num_targets))
        
        result = {'energy': energy}
        
        # Process vector features for force prediction
        if 1 in features and self.l_max >= 1:
            x_vector = features[1]  # Shape: (n_bonds, 3)
            # Linear projection to force vectors (l=1 -> l=1)
            force_vectors = self.force_transforms[1](x_vector)
            
            # Aggregate forces to atoms
            forces = tf.math.unsorted_segment_sum(force_vectors, idnb_i, n_atoms)
            result['forces'] = forces  # Shape: (n_atoms, 3)
        
        return result

    def compute_forces(self, energy, R):
        """
        Compute forces as negative gradient of energy with respect to positions.
        
        Parameters
        ----------
        energy : tf.Tensor
            Energy predictions of shape (n_atoms,)
        R : tf.Tensor
            Atomic positions of shape (n_atoms, 3)
            
        Returns
        -------
        tf.Tensor
            Forces of shape (n_atoms, 3)
        """
        with tf.GradientTape() as tape:
            tape.watch(R)
            # Reshape energy to scalar for gradient computation
            energy_scalar = tf.reduce_sum(energy)
        
        forces = -tape.gradient(energy_scalar, R)
        return forces

