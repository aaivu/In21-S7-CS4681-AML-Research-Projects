"""
Example script demonstrating E(3)-equivariant DimeNet++ usage.

This script shows how to use the equivariant DimeNet++ model for molecular property prediction.
"""

import numpy as np
import tensorflow as tf
from dimenet.model.dimenet_pp_equivariant import DimeNetPPEquivariant
from dimenet.model.layers.equivariant_utils import create_irreducible_features
from dimenet.model.layers.tensor_algebra import TensorProductLayer, SphericalHarmonicsLayer


def create_sample_molecule():
    """Create a sample molecule for demonstration."""
    # Water molecule (H2O)
    Z = np.array([8, 1, 1])  # O, H, H
    R = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H
        [-0.24, 0.93, 0.0]    # H
    ])
    
    # Create neighbor indices (simplified)
    idnb_i = np.array([0, 0, 1, 1, 2, 2])  # Central atoms
    idnb_j = np.array([1, 2, 0, 2, 0, 1])  # Neighbor atoms
    
    # Create batch segmentation
    batch_seg = np.array([0, 0, 0])  # All atoms in same batch
    
    # Create triple interaction indices (simplified)
    id_expand_kj = np.array([0, 1, 2, 3, 4, 5])
    id_reduce_ji = np.array([0, 0, 1, 1, 2, 2])
    id3dnb_i = np.array([0, 0, 1, 1, 2, 2])
    id3dnb_j = np.array([1, 2, 0, 2, 0, 1])
    id3dnb_k = np.array([2, 1, 2, 0, 1, 0])
    
    return {
        'Z': tf.constant(Z, dtype=tf.int32),
        'R': tf.constant(R, dtype=tf.float32),
        'batch_seg': tf.constant(batch_seg, dtype=tf.int32),
        'idnb_i': tf.constant(idnb_i, dtype=tf.int32),
        'idnb_j': tf.constant(idnb_j, dtype=tf.int32),
        'id_expand_kj': tf.constant(id_expand_kj, dtype=tf.int32),
        'id_reduce_ji': tf.constant(id_reduce_ji, dtype=tf.int32),
        'id3dnb_i': tf.constant(id3dnb_i, dtype=tf.int32),
        'id3dnb_j': tf.constant(id3dnb_j, dtype=tf.int32),
        'id3dnb_k': tf.constant(id3dnb_k, dtype=tf.int32)
    }


def demonstrate_equivariant_features():
    """Demonstrate equivariant feature creation."""
    print("=== Demonstrating Equivariant Features ===")
    
    # Create sample data
    scalar_features = tf.random.normal((10, 64))  # 10 bonds, 64 features each
    vector_features = tf.random.normal((10, 3))   # 10 bonds, 3D vectors
    
    # Create irreducible features
    features = create_irreducible_features(
        scalar_features=scalar_features,
        vector_features=vector_features,
        l_max=1
    )
    
    print(f"Scalar features (l=0): {features[0].shape}")
    print(f"Vector features (l=1): {features[1].shape}")
    
    # Demonstrate rotation equivariance
    # Create rotation matrix
    angle = np.pi / 4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Rotate vector features
    vector_features_rotated = tf.matmul(vector_features, rotation_matrix.T)
    
    # Create rotated features
    features_rotated = create_irreducible_features(
        scalar_features=scalar_features,
        vector_features=vector_features_rotated,
        l_max=1
    )
    
    print(f"Original vector norm: {tf.linalg.norm(features[1], axis=-1)[:3]}")
    print(f"Rotated vector norm: {tf.linalg.norm(features_rotated[1], axis=-1)[:3]}")
    print("Vector norms should be the same (rotation invariant)")


def demonstrate_tensor_products():
    """Demonstrate tensor product operations."""
    print("\n=== Demonstrating Tensor Products ===")
    
    # Create sample features
    features1 = {
        0: tf.random.normal((5, 1)),  # l=0 (scalar)
        1: tf.random.normal((5, 3))   # l=1 (vector)
    }
    
    features2 = {
        0: tf.random.normal((5, 1)),  # l=0 (scalar)
        1: tf.random.normal((5, 3))   # l=1 (vector)
    }
    
    # Create tensor product layer
    tensor_product = TensorProductLayer(l_max=2)
    
    # Compute tensor products
    result = tensor_product(features1, features2)
    
    print(f"Input features1: {[(l, f.shape) for l, f in features1.items()]}")
    print(f"Input features2: {[(l, f.shape) for l, f in features2.items()]}")
    print(f"Tensor product result: {[(l, f.shape) for l, f in result.items()]}")


def demonstrate_spherical_harmonics():
    """Demonstrate spherical harmonics computation."""
    print("\n=== Demonstrating Spherical Harmonics ===")
    
    # Create sample direction vectors
    directions = tf.constant([
        [1.0, 0.0, 0.0],  # x-axis
        [0.0, 1.0, 0.0],  # y-axis
        [0.0, 0.0, 1.0],  # z-axis
        [1.0, 1.0, 0.0]   # diagonal
    ])
    
    # Create spherical harmonics layer
    sph_harm = SphericalHarmonicsLayer(l_max=2)
    
    # Compute spherical harmonics
    harmonics = sph_harm(directions)
    
    print(f"Direction vectors shape: {directions.shape}")
    for l, y_lm in harmonics.items():
        print(f"Y_l^{l} shape: {y_lm.shape}")
        print(f"Y_l^{l} values: {y_lm.numpy()[:2]}")  # Show first 2 directions


def demonstrate_equivariant_model():
    """Demonstrate the equivariant DimeNet++ model."""
    print("\n=== Demonstrating Equivariant DimeNet++ Model ===")
    
    # Create model
    model = DimeNetPPEquivariant(
        emb_size=64,
        out_emb_size=128,
        int_emb_size=32,
        basis_emb_size=8,
        num_blocks=2,
        num_spherical=5,
        num_radial=6,
        l_max=1,
        num_targets=1,
        predict_forces=True
    )
    
    # Create sample input
    inputs = create_sample_molecule()
    
    # Forward pass
    outputs = model(inputs)
    
    print(f"Model input shapes:")
    for key, value in inputs.items():
        print(f"  {key}: {value.shape}")
    
    print(f"Model output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test equivariance
    print("\nTesting rotation equivariance...")
    
    # Create rotation matrix
    angle = np.pi / 3
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Rotate positions
    R_rotated = tf.matmul(inputs['R'], rotation_matrix.T)
    inputs_rotated = inputs.copy()
    inputs_rotated['R'] = R_rotated
    
    # Forward pass with rotated input
    outputs_rotated = model(inputs_rotated)
    
    # Check energy invariance
    energy_diff = tf.abs(outputs['energy'] - outputs_rotated['energy'])
    print(f"Energy difference after rotation: {energy_diff.numpy()}")
    print("Energy should be invariant (difference ≈ 0)")
    
    # Check force equivariance
    if 'forces' in outputs and 'forces' in outputs_rotated:
        forces_rotated_expected = tf.matmul(outputs['forces'], rotation_matrix.T)
        force_diff = tf.linalg.norm(outputs_rotated['forces'] - forces_rotated_expected)
        print(f"Force equivariance error: {force_diff.numpy()}")
        print("Forces should be equivariant (error ≈ 0)")


def main():
    """Main demonstration function."""
    print("E(3)-Equivariant DimeNet++ Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate individual components
        demonstrate_equivariant_features()
        demonstrate_tensor_products()
        demonstrate_spherical_harmonics()
        demonstrate_equivariant_model()
        
        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print("\nKey features demonstrated:")
        print("1. Irreducible tensor feature creation")
        print("2. Tensor product operations")
        print("3. Spherical harmonics computation")
        print("4. E(3)-equivariant model forward pass")
        print("5. Rotation invariance/equivariance properties")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This might be due to missing dependencies or configuration issues.")


if __name__ == '__main__':
    main()


