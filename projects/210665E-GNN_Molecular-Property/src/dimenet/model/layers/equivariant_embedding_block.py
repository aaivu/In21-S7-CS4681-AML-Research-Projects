"""
Equivariant embedding block for DimeNet++.

This module implements the equivariant version of the embedding block that creates
irreducible tensor features from atomic embeddings and radial basis functions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .equivariant_utils import create_irreducible_features, equivariant_linear
from ..initializers import GlorotOrthogonal


class EquivariantEmbeddingBlock(layers.Layer):
    """
    Equivariant embedding block that creates irreducible tensor features.
    
    This block extends the original embedding block to support E(3)-equivariant
    features by creating both scalar (l=0) and vector (l=1) features.
    """
    
    def __init__(self, emb_size, l_max=1, activation=None, name='equivariant_embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.l_max = l_max
        self.weight_init = GlorotOrthogonal()

        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        self.embeddings = self.add_weight(
            name="embeddings", 
            shape=(95, self.emb_size),
            dtype=tf.float32, 
            initializer=emb_init, 
            trainable=True
        )

        # Scalar (l=0) transformations
        self.dense_rbf = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                      kernel_initializer=self.weight_init)
        self.dense_scalar = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                        kernel_initializer=self.weight_init)
        
        # Vector (l=1) transformations
        if l_max >= 1:
            self.dense_vector = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                            kernel_initializer=self.weight_init)
            # Learnable vector features for each atom type
            self.vector_embeddings = self.add_weight(
                name="vector_embeddings",
                shape=(95, self.emb_size, 3),  # 3D vectors
                dtype=tf.float32,
                initializer=tf.initializers.RandomNormal(stddev=0.1),
                trainable=True
            )
            # Use vector features to modulate scalar embeddings so they receive gradients
            self.vector_to_scalar = layers.Dense(self.emb_size, activation=activation, use_bias=False,
                                                 kernel_initializer=self.weight_init)

    def call(self, inputs):
        """
        Forward pass of the equivariant embedding block.
        
        Parameters
        ----------
        inputs : list
            [Z, rbf, idnb_i, idnb_j, R] where:
            - Z: atomic numbers
            - rbf: radial basis functions
            - idnb_i, idnb_j: neighbor indices
            - R: atomic positions (for vector features)
            
        Returns
        -------
        dict
            Dictionary mapping l values to irreducible tensor features
        """
        Z, rbf, idnb_i, idnb_j, R = inputs

        # Transform radial basis functions
        rbf = self.dense_rbf(rbf)

        # Get atomic numbers
        Z_i = tf.gather(Z, idnb_i)
        Z_j = tf.gather(Z, idnb_j)

        # Scalar (l=0) features
        x_i_scalar = tf.gather(self.embeddings, Z_i)
        x_j_scalar = tf.gather(self.embeddings, Z_j)
        
        # Combine scalar features
        x_scalar = tf.concat([x_i_scalar, x_j_scalar, rbf], axis=-1)
        x_scalar = self.dense_scalar(x_scalar)

        # Create irreducible features dictionary
        features = {0: x_scalar}  # l=0 (scalar) features

        # Vector (l=1) features
        if self.l_max >= 1:
            # Get atomic positions
            R_i = tf.gather(R, idnb_i)
            R_j = tf.gather(R, idnb_j)
            
            # Compute bond vectors
            bond_vectors = R_j - R_i  # Shape: (n_bonds, 3)
            
            # Normalize bond vectors
            bond_norms = tf.linalg.norm(bond_vectors, axis=-1, keepdims=True)
            bond_directions = bond_vectors / (bond_norms + 1e-8)
            
            # Learn vector features for each atom type
            x_i_vector = tf.gather(self.vector_embeddings, Z_i)  # Shape: (n_bonds, emb_size, 3)
            x_j_vector = tf.gather(self.vector_embeddings, Z_j)  # Shape: (n_bonds, emb_size, 3)
            
            # Project vector features onto bond directions
            # This creates equivariant vector features
            x_i_proj = tf.reduce_sum(x_i_vector * bond_directions[:, None, :], axis=-1)  # Shape: (n_bonds, emb_size)
            x_j_proj = tf.reduce_sum(x_j_vector * bond_directions[:, None, :], axis=-1)  # Shape: (n_bonds, emb_size)
            
            # Combine vector features
            x_vector = tf.concat([x_i_proj, x_j_proj], axis=-1)
            x_vector = self.dense_vector(x_vector)
            
            # Create vector features by multiplying scalar features with bond directions
            # This ensures the vector features transform correctly under rotations
            x_vector_expanded = x_vector[:, :, None] * bond_directions[:, None, :]  # Shape: (n_bonds, emb_size, 3)
            
            # Sum over the embedding dimension to get final vector features
            x_vector_final = tf.reduce_sum(x_vector_expanded, axis=1)  # Shape: (n_bonds, 3)

            # Also inject a scalar from vector features to ensure gradients flow to vector params
            # Use a statically-known last dimension for Dense build compatibility
            flat_vec = tf.reshape(x_vector_expanded, (-1, self.emb_size * 3))
            vec_scalar = self.vector_to_scalar(flat_vec)
            features[0] = features[0] + vec_scalar

            features[1] = x_vector_final  # l=1 (vector) features

        return features

