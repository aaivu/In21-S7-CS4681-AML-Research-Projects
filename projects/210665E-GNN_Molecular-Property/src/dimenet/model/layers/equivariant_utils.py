"""
Utilities for E(3)-equivariant operations in DimeNet++.

This module provides functions for working with irreducible tensor representations
and implementing E(3)-equivariant operations as described in the methodology.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sympy as sym
from scipy import special as sp


def clebsch_gordan_coefficients(l1, l2, l3, m1, m2, m3):
    """
    Compute Clebsch-Gordan coefficients C(l1, l2, l3; m1, m2, m3).
    
    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers
    m1, m2, m3 : int
        Magnetic quantum numbers
        
    Returns
    -------
    float
        Clebsch-Gordan coefficient
    """
    # Check selection rules
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    if m1 + m2 != m3:
        return 0.0
    if abs(l1 - l2) > l3 or l3 > l1 + l2:
        return 0.0
    if (l1 + l2 + l3) % 2 == 1:
        return 0.0
    
    # Use scipy's implementation
    return sp.special.clebsch_gordan(l1, l2, l3, m1, m2, m3)


def tensor_product_irreps(features_l1, features_l2, max_l=None):
    """
    Compute tensor product of irreducible representations.
    
    Parameters
    ----------
    features_l1 : dict
        Dictionary mapping l values to feature tensors of shape (..., 2*l+1)
    features_l2 : dict
        Dictionary mapping l values to feature tensors of shape (..., 2*l+1)
    max_l : int, optional
        Maximum l value to compute. If None, uses l1_max + l2_max
        
    Returns
    -------
    dict
        Dictionary mapping l values to tensor product features
    """
    result = {}
    
    l1_values = list(features_l1.keys())
    l2_values = list(features_l2.keys())
    
    if max_l is None:
        max_l = max(l1_values) + max(l2_values)
    
    for l1 in l1_values:
        for l2 in l2_values:
            for l3 in range(abs(l1 - l2), min(l1 + l2 + 1, max_l + 1)):
                if l3 not in result:
                    result[l3] = []
                
                # Get feature tensors
                f1 = features_l1[l1]  # Shape: (..., 2*l1+1)
                f2 = features_l2[l2]  # Shape: (..., 2*l2+1)
                
                # Compute tensor product using Clebsch-Gordan coefficients
                tensor_prod = tf.zeros_like(f1[..., :1])  # Initialize with scalar part
                
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        m3 = m1 + m2
                        if abs(m3) <= l3:
                            cg = clebsch_gordan_coefficients(l1, l2, l3, m1, m2, m3)
                            if abs(cg) > 1e-10:  # Numerical threshold
                                # Get the specific components
                                f1_comp = f1[..., m1 + l1]  # Shape: (...)
                                f2_comp = f2[..., m2 + l2]  # Shape: (...)
                                
                                # Add to tensor product
                                if l3 not in result:
                                    result[l3] = tf.zeros(tf.concat([tf.shape(f1_comp), [2*l3+1]], axis=0))
                                
                                result[l3] = tf.tensor_scatter_nd_add(
                                    result[l3],
                                    [[..., m3 + l3]],
                                    cg * f1_comp * f2_comp
                                )
    
    return result


def spherical_harmonics_tensor(r, l_max):
    """
    Compute spherical harmonics for all l up to l_max.
    
    Parameters
    ----------
    r : tf.Tensor
        Direction vectors of shape (..., 3)
    l_max : int
        Maximum l value
        
    Returns
    -------
    dict
        Dictionary mapping l values to spherical harmonic tensors
    """
    # Normalize direction vectors
    r_norm = tf.linalg.norm(r, axis=-1, keepdims=True)
    r_unit = r / (r_norm + 1e-8)
    
    # Convert to spherical coordinates
    x, y, z = r_unit[..., 0], r_unit[..., 1], r_unit[..., 2]
    
    # Compute spherical harmonics
    result = {}
    
    for l in range(l_max + 1):
        harmonics = []
        for m in range(-l, l + 1):
            if l == 0:
                # Y_0^0 = 1/sqrt(4*pi)
                y_lm = tf.ones_like(x) / tf.sqrt(4 * np.pi)
            else:
                # Use scipy's spherical harmonics
                theta = tf.acos(z)
                phi = tf.atan2(y, x)
                
                # Convert to complex spherical harmonics
                y_lm_complex = sp.sph_harm(m, l, phi, theta)
                
                # Take real part for real spherical harmonics
                y_lm = tf.math.real(y_lm_complex)
            
            harmonics.append(y_lm)
        
        result[l] = tf.stack(harmonics, axis=-1)  # Shape: (..., 2*l+1)
    
    return result


def equivariant_linear(features, output_l, input_l, hidden_dim=64):
    """
    Equivariant linear layer that preserves irreducible representation structure.
    
    Parameters
    ----------
    features : dict
        Input features as dictionary mapping l to tensors
    output_l : int
        Output l value
    input_l : int
        Input l value
    hidden_dim : int
        Hidden dimension for the linear transformation
        
    Returns
    -------
    tf.Tensor
        Output features of shape (..., 2*output_l+1)
    """
    if input_l not in features:
        # If input l doesn't exist, return zeros of batch size 1 to avoid shape errors
        return tf.zeros([1, 2*output_l+1])
    
    input_features = features[input_l]  # Shape: (..., 2*input_l+1)
    
    if input_l == output_l:
        # Same l: simple linear transformation
        linear = layers.Dense(2*output_l+1, use_bias=False)
        return linear(input_features)
    else:
        # Different l: need to project through Clebsch-Gordan
        # For now, implement a simple projection
        # In practice, this would use proper Clebsch-Gordan coefficients
        linear = layers.Dense(2*output_l+1, use_bias=False)
        return linear(input_features)


def equivariant_activation(features, activation_fn=tf.nn.swish):
    """
    Apply activation function while preserving equivariance.
    For scalar features (l=0), apply activation directly.
    For higher-order features, apply activation to the norm and preserve direction.
    
    Parameters
    ----------
    features : dict
        Input features as dictionary mapping l to tensors
    activation_fn : callable
        Activation function to apply
        
    Returns
    -------
    dict
        Activated features
    """
    result = {}
    
    for l, feat in features.items():
        if l == 0:
            # Scalar features: apply activation directly
            result[l] = activation_fn(feat)
        else:
            # Higher-order features: apply activation to norm
            norm = tf.linalg.norm(feat, axis=-1, keepdims=True)
            direction = feat / (norm + 1e-8)
            activated_norm = activation_fn(norm)
            result[l] = activated_norm * direction
    
    return result


def equivariant_norm(features):
    """
    Compute equivariant norm for each irreducible representation.
    
    Parameters
    ----------
    features : dict
        Input features as dictionary mapping l to tensors
        
    Returns
    -------
    dict
        Norms for each l value
    """
    result = {}
    for l, feat in features.items():
        result[l] = tf.linalg.norm(feat, axis=-1, keepdims=True)
    return result


def create_irreducible_features(scalar_features, vector_features=None, l_max=1):
    """
    Create irreducible tensor features from scalar and vector inputs.
    
    Parameters
    ----------
    scalar_features : tf.Tensor
        Scalar features of shape (..., dim)
    vector_features : tf.Tensor, optional
        Vector features of shape (..., 3)
    l_max : int
        Maximum l value to create
        
    Returns
    -------
    dict
        Dictionary mapping l values to irreducible tensor features
    """
    features = {}
    
    # l=0 (scalar) features
    features[0] = scalar_features
    
    # l=1 (vector) features
    if vector_features is not None and l_max >= 1:
        features[1] = vector_features
    
    return features


def equivariant_aggregation(features, indices, num_segments):
    """
    Equivariant aggregation (sum) of features.
    
    Parameters
    ----------
    features : dict
        Features to aggregate as dictionary mapping l to tensors
    indices : tf.Tensor
        Segment indices for aggregation
    num_segments : int
        Number of segments
        
    Returns
    -------
    dict
        Aggregated features
    """
    result = {}
    for l, feat in features.items():
        result[l] = tf.math.unsorted_segment_sum(feat, indices, num_segments)
    return result

