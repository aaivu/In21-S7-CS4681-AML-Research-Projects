"""
Tensor algebra operations for E(3)-equivariant neural networks.

This module implements tensor products, Clebsch-Gordan coefficients, and other
operations needed for working with irreducible representations of SO(3).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy import special as sp
import sympy as sym


class ClebschGordanCoefficients:
    """
    Precomputed Clebsch-Gordan coefficients for efficient tensor products.
    """
    
    def __init__(self, l_max=3):
        self.l_max = l_max
        self.coefficients = {}
        self._precompute_coefficients()
    
    def _precompute_coefficients(self):
        """Precompute Clebsch-Gordan coefficients up to l_max."""
        for l1 in range(self.l_max + 1):
            for l2 in range(self.l_max + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, self.l_max + 1)):
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            m3 = m1 + m2
                            if abs(m3) <= l3:
                                cg = sp.special.clebsch_gordan(l1, l2, l3, m1, m2, m3)
                                if abs(cg) > 1e-10:  # Numerical threshold
                                    key = (l1, l2, l3, m1, m2, m3)
                                    self.coefficients[key] = float(cg)
    
    def get_coefficient(self, l1, l2, l3, m1, m2, m3):
        """Get Clebsch-Gordan coefficient."""
        key = (l1, l2, l3, m1, m2, m3)
        return self.coefficients.get(key, 0.0)


class TensorProductLayer(layers.Layer):
    """
    Layer that computes tensor products of irreducible representations.
    """
    
    def __init__(self, l_max=3, name='tensor_product', **kwargs):
        super().__init__(name=name, **kwargs)
        self.l_max = l_max
        self.cg_coeffs = ClebschGordanCoefficients(l_max)
    
    def call(self, features1, features2):
        """
        Compute tensor product of two sets of irreducible features.
        
        Parameters
        ----------
        features1 : dict
            First set of features mapping l to tensors
        features2 : dict
            Second set of features mapping l to tensors
            
        Returns
        -------
        dict
            Tensor product features
        """
        result = {}
        
        for l1, f1 in features1.items():
            for l2, f2 in features2.items():
                for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, self.l_max + 1)):
                    if l3 not in result:
                        result[l3] = []
                    
                    # Compute tensor product for this (l1, l2, l3) combination
                    tensor_prod = self._compute_tensor_product(f1, f2, l1, l2, l3)
                    result[l3].append(tensor_prod)
        
        # Sum over all contributions to each l3
        for l3 in result:
            result[l3] = tf.add_n(result[l3])
        
        return result
    
    def _compute_tensor_product(self, f1, f2, l1, l2, l3):
        """Compute tensor product for specific l values."""
        # f1 shape: (..., 2*l1+1)
        # f2 shape: (..., 2*l2+1)
        # result shape: (..., 2*l3+1)
        
        result = tf.zeros(tf.concat([tf.shape(f1)[:-1], [2*l3+1]], axis=0))
        
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m3 = m1 + m2
                if abs(m3) <= l3:
                    cg = self.cg_coeffs.get_coefficient(l1, l2, l3, m1, m2, m3)
                    if abs(cg) > 1e-10:
                        # Get specific components
                        f1_comp = f1[..., m1 + l1]  # Shape: (...)
                        f2_comp = f2[..., m2 + l2]  # Shape: (...)
                        
                        # Add to result
                        result = tf.tensor_scatter_nd_add(
                            result,
                            [[..., m3 + l3]],
                            cg * f1_comp * f2_comp
                        )
        
        return result


class SphericalHarmonicsLayer(layers.Layer):
    """
    Layer that computes spherical harmonics for direction vectors.
    """
    
    def __init__(self, l_max=3, name='spherical_harmonics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.l_max = l_max
        self._precompute_harmonics()
    
    def _precompute_harmonics(self):
        """Precompute spherical harmonic formulas."""
        self.harmonic_funcs = {}
        
        for l in range(self.l_max + 1):
            self.harmonic_funcs[l] = []
            for m in range(-l, l + 1):
                if l == 0:
                    # Y_0^0 = 1/sqrt(4*pi)
                    self.harmonic_funcs[l].append(lambda x, y, z: tf.ones_like(x) / tf.sqrt(4 * np.pi))
                else:
                    # Use scipy's spherical harmonics
                    def make_harmonic_func(l_val, m_val):
                        def harmonic_func(x, y, z):
                            # Convert to spherical coordinates
                            r = tf.sqrt(x**2 + y**2 + z**2)
                            theta = tf.acos(z / (r + 1e-8))
                            phi = tf.atan2(y, x)
                            
                            # Compute spherical harmonic
                            y_lm = sp.sph_harm(m_val, l_val, phi, theta)
                            return tf.math.real(y_lm)
                        return harmonic_func
                    
                    self.harmonic_funcs[l].append(make_harmonic_func(l, m))
    
    def call(self, directions):
        """
        Compute spherical harmonics for direction vectors.
        
        Parameters
        ----------
        directions : tf.Tensor
            Direction vectors of shape (..., 3)
            
        Returns
        -------
        dict
            Dictionary mapping l to spherical harmonic tensors
        """
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        
        result = {}
        for l in range(self.l_max + 1):
            harmonics = []
            for m in range(-l, l + 1):
                y_lm = self.harmonic_funcs[l][m + l](x, y, z)
                harmonics.append(y_lm)
            result[l] = tf.stack(harmonics, axis=-1)  # Shape: (..., 2*l+1)
        
        return result


class EquivariantConvolutionLayer(layers.Layer):
    """
    E(3)-equivariant convolution layer using spherical harmonics.
    """
    
    def __init__(self, emb_size, l_max=3, name='equivariant_convolution', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.l_max = l_max
        
        # Spherical harmonics layer
        self.sph_harm = SphericalHarmonicsLayer(l_max)
        
        # Radial function (learnable)
        self.radial_fn = layers.Dense(emb_size, activation=tf.nn.swish, use_bias=True)
        
        # Output projections for each l
        self.output_projections = {}
        for l in range(l_max + 1):
            self.output_projections[l] = layers.Dense(2*l+1, use_bias=False)
    
    def call(self, features, directions, distances):
        """
        Apply equivariant convolution.
        
        Parameters
        ----------
        features : dict
            Input features mapping l to tensors
        directions : tf.Tensor
            Direction vectors of shape (..., 3)
        distances : tf.Tensor
            Distances of shape (...)
            
        Returns
        -------
        dict
            Convolved features
        """
        # Compute spherical harmonics
        sph_harm = self.sph_harm(directions)
        
        # Compute radial function
        radial = self.radial_fn(distances[..., None])  # Shape: (..., emb_size)
        
        result = {}
        for l, feat in features.items():
            if l in sph_harm:
                # Apply convolution: R(r) * Y_l^m(r̂) * f_l^m
                radial_expanded = radial[..., None]  # Shape: (..., emb_size, 1)
                sph_harm_l = sph_harm[l]  # Shape: (..., 2*l+1)
                
                # Element-wise multiplication
                convolved = feat * radial_expanded * sph_harm_l[..., None, :]
                
                # Project to output
                convolved = self.output_projections[l](convolved)
                result[l] = convolved
        
        return result


class SteerableMLP(layers.Layer):
    """
    Steerable MLP that preserves equivariance.
    """
    
    def __init__(self, hidden_dims, l_max=3, activation=tf.nn.swish, name='steerable_mlp', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_dims = hidden_dims
        self.l_max = l_max
        self.activation = activation
        
        # Create MLP layers for each l
        self.mlp_layers = {}
        for l in range(l_max + 1):
            self.mlp_layers[l] = []
            for hidden_dim in hidden_dims:
                self.mlp_layers[l].append(
                    layers.Dense(hidden_dim, activation=activation, use_bias=True)
                )
    
    def call(self, features):
        """
        Apply steerable MLP.
        
        Parameters
        ----------
        features : dict
            Input features mapping l to tensors
            
        Returns
        -------
        dict
            Transformed features
        """
        result = {}
        
        for l, feat in features.items():
            if l in self.mlp_layers:
                x = feat
                for layer in self.mlp_layers[l]:
                    x = layer(x)
                result[l] = x
        
        return result


class EquivariantTripleInteraction(layers.Layer):
    """
    Equivariant triple interaction using tensor products.
    """
    
    def __init__(self, emb_size, l_max=3, name='equivariant_triple', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.l_max = l_max
        
        # Tensor product layer
        self.tensor_product = TensorProductLayer(l_max)
        
        # Steerable MLP for processing
        self.steerable_mlp = SteerableMLP([emb_size], l_max)
    
    def call(self, features_i, features_j, features_k, directions_ij, directions_jk):
        """
        Compute equivariant triple interaction.
        
        Parameters
        ----------
        features_i, features_j, features_k : dict
            Features for atoms i, j, k
        directions_ij, directions_jk : tf.Tensor
            Direction vectors for bonds ij and jk
            
        Returns
        -------
        dict
            Triple interaction features
        """
        # Compute tensor product of features j and k
        features_jk = self.tensor_product(features_j, features_k)
        
        # Apply steerable MLP
        features_jk = self.steerable_mlp(features_jk)
        
        # Combine with features i (this would need more sophisticated logic
        # in a full implementation)
        result = {}
        for l in range(self.l_max + 1):
            if l in features_i and l in features_jk:
                result[l] = features_i[l] + features_jk[l]
            elif l in features_i:
                result[l] = features_i[l]
            elif l in features_jk:
                result[l] = features_jk[l]
        
        return result


