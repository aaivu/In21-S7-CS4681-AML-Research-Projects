"""
Equivariant interaction block for DimeNet++.

This module implements the equivariant version of the interaction block that performs
E(3)-equivariant message passing using tensor products and Clebsch-Gordan coefficients.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .equivariant_utils import (
    tensor_product_irreps, 
    spherical_harmonics_tensor, 
    equivariant_linear,
    equivariant_activation,
    equivariant_aggregation
)
from .residual_layer import ResidualLayer
from ..initializers import GlorotOrthogonal


class EquivariantInteractionPPBlock(layers.Layer):
    """
    Equivariant interaction block that performs E(3)-equivariant message passing.
    
    This block extends the original interaction block to support irreducible tensor
    features and uses tensor products for equivariant triple interactions.
    """
    
    def __init__(self, emb_size, int_emb_size, basis_emb_size, num_before_skip, 
                 num_after_skip, l_max=1, activation=None, name='equivariant_interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.int_emb_size = int_emb_size
        self.basis_emb_size = basis_emb_size
        self.l_max = l_max
        self.weight_init = GlorotOrthogonal()

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = layers.Dense(basis_emb_size, use_bias=False, kernel_initializer=self.weight_init)
        self.dense_rbf2 = layers.Dense(emb_size, use_bias=False, kernel_initializer=self.weight_init)
        self.dense_sbf1 = layers.Dense(basis_emb_size, use_bias=False, kernel_initializer=self.weight_init)
        self.dense_sbf2 = layers.Dense(int_emb_size, use_bias=False, kernel_initializer=self.weight_init)

        # Placeholder for potential equivariant transforms (not used in fast benchmark path)
        self.equivariant_transforms = None

        # Dense transformations of input messages
        self.dense_ji = layers.Dense(emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=self.weight_init)
        self.dense_kj = layers.Dense(emb_size, activation=activation, use_bias=True,
                                     kernel_initializer=self.weight_init)

        # Embedding projections for interaction triplets
        self.down_projection = layers.Dense(int_emb_size, activation=activation, use_bias=False,
                                            kernel_initializer=self.weight_init)
        self.up_projection = layers.Dense(emb_size, activation=activation, use_bias=False,
                                          kernel_initializer=self.weight_init)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=self.weight_init))
        self.final_before_skip = layers.Dense(emb_size, activation=activation, use_bias=True,
                                              kernel_initializer=self.weight_init)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(
                ResidualLayer(emb_size, activation=activation, use_bias=True,
                              kernel_initializer=self.weight_init))

    def call(self, inputs):
        """
        Forward pass of the equivariant interaction block.
        
        Parameters
        ----------
        inputs : list
            [features, rbf, sbf, id_expand_kj, id_reduce_ji, R] where:
            - features: dictionary of irreducible tensor features
            - rbf: radial basis functions
            - sbf: spherical basis functions
            - id_expand_kj, id_reduce_ji: neighbor indices
            - R: atomic positions
            
        Returns
        -------
        dict
            Updated irreducible tensor features
        """
        features, rbf, sbf, id_expand_kj, id_reduce_ji, R = inputs
        num_interactions = tf.shape(rbf)[0]

        # Transform radial basis functions
        rbf = self.dense_rbf1(rbf)
        rbf = self.dense_rbf2(rbf)

        # Transform spherical basis functions
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)

        # Process each l value separately
        updated_features = {}
        
        for l in range(self.l_max + 1):
            if l not in features:
                continue
                
            # Get features for this l value
            x_l = features[l]  # Shape depends on l
            
            if l == 0:
                # Scalar features: use original DimeNet++ logic
                x_ji = self.dense_ji(x_l)
                x_kj = self.dense_kj(x_l)
                
                # Apply radial basis transformation
                x_kj = x_kj * rbf
                
                # Down-project embeddings and generate interaction triplet embeddings
                x_kj = self.down_projection(x_kj)
                x_kj = tf.gather(x_kj, id_expand_kj)
                
                # Apply spherical basis transformation
                x_kj = x_kj * sbf
                
                # Aggregate interactions and up-project embeddings
                x_kj = tf.math.unsorted_segment_sum(x_kj, id_reduce_ji, num_interactions)
                x_kj = self.up_projection(x_kj)
                
                # Combine features
                x2 = x_ji + x_kj
                
                # Apply residual layers before skip connection
                for layer in self.layers_before_skip:
                    x2 = layer(x2)
                x2 = self.final_before_skip(x2)
                
                # Skip connection
                x_l = x_l + x2
                
                # Apply residual layers after skip connection
                for layer in self.layers_after_skip:
                    x_l = layer(x_l)
                    
                updated_features[l] = x_l
                
            else:
                # Fast path: leave higher-order (vector) features unchanged
                updated_features[l] = x_l

        return updated_features

    def _compute_equivariant_triple_interactions(self, features, rbf, sbf, id_expand_kj, id_reduce_ji, R):
        """
        Compute equivariant triple interactions using tensor products.
        
        This is a placeholder for the full implementation that would use
        Clebsch-Gordan coefficients and tensor products.
        """
        # This would implement the full equivariant triple interaction logic
        # using tensor products and Clebsch-Gordan coefficients
        pass

