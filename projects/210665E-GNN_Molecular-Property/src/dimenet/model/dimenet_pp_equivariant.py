"""
E(3)-equivariant DimeNet++ model.

This module implements the E(3)-equivariant version of DimeNet++ that extends
the original model to support irreducible tensor features and equivariant operations.
"""

import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.equivariant_embedding_block import EquivariantEmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.equivariant_interaction_pp_block import EquivariantInteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .layers.equivariant_output_pp_block import EquivariantOutputPPBlock
from .layers.equivariant_utils import equivariant_aggregation
from .activations import swish


class DimeNetPPEquivariant(tf.keras.Model):
    """
    E(3)-equivariant DimeNet++ model.

    This model extends DimeNet++ to support E(3)-equivariant operations by:
    1. Using irreducible tensor features (scalar and vector)
    2. Implementing equivariant convolutional filters
    3. Using tensor products for triple interactions
    4. Ensuring output invariance and force equivariance

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    l_max
        Maximum l value for irreducible representations
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initialization method for the output layer (last layer in output block)
    predict_forces
        Whether to predict forces in addition to energies
    """

    def __init__(
            self, emb_size, out_emb_size, int_emb_size, basis_emb_size,
            num_blocks, num_spherical, num_radial,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=12,
            l_max=1, activation=swish, extensive=True, output_init='zeros',
            predict_forces=False, name='dimenet_pp_equivariant', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.l_max = l_max
        self.predict_forces = predict_forces

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Equivariant embedding and first output block
        self.output_blocks = []
        self.emb_block = EquivariantEmbeddingBlock(
            emb_size, l_max=l_max, activation=activation)
        self.output_blocks.append(
            EquivariantOutputPPBlock(
                emb_size, out_emb_size, num_dense_output, num_targets,
                l_max=l_max, activation=activation, output_init=output_init))

        # Equivariant interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                EquivariantInteractionPPBlock(
                    emb_size, int_emb_size, basis_emb_size, num_before_skip,
                    num_after_skip, l_max=l_max, activation=activation))
            self.output_blocks.append(
                EquivariantOutputPPBlock(
                    emb_size, out_emb_size, num_dense_output, num_targets,
                    l_max=l_max, activation=activation, output_init=output_init))

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        """Calculate interatomic distances."""
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets."""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs):
        """
        Forward pass of the equivariant DimeNet++ model.
        
        Parameters
        ----------
        inputs : dict
            Dictionary containing:
            - Z: atomic numbers
            - R: atomic positions
            - batch_seg: batch segmentation
            - idnb_i, idnb_j: neighbor indices
            - id_expand_kj, id_reduce_ji: triple interaction indices
            - id3dnb_i, id3dnb_j, id3dnb_k: angle calculation indices
            
        Returns
        -------
        dict
            Dictionary containing 'energy' and optionally 'forces'
        """
        Z, R = inputs['Z'], inputs['R']
        batch_seg = inputs['batch_seg']
        idnb_i, idnb_j = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Equivariant embedding block
        features = self.emb_block([Z, rbf, idnb_i, idnb_j, R])
        
        # First output block
        output = self.output_blocks[0]([features, rbf, idnb_i, n_atoms, R])
        energy = output['energy']
        forces = output.get('forces', None)

        # Interaction blocks
        for i in range(self.num_blocks):
            features = self.int_blocks[i]([features, rbf, sbf, id_expand_kj, id_reduce_ji, R])
            output = self.output_blocks[i+1]([features, rbf, idnb_i, n_atoms, R])
            energy += output['energy']
            if forces is not None and 'forces' in output:
                forces += output['forces']

        # Aggregate over batch
        if self.extensive:
            energy = tf.math.segment_sum(energy, batch_seg)
            if forces is not None:
                forces = tf.math.segment_sum(forces, batch_seg)
        else:
            energy = tf.math.segment_mean(energy, batch_seg)
            if forces is not None:
                forces = tf.math.segment_mean(forces, batch_seg)

        result = {'energy': energy}
        if forces is not None:
            result['forces'] = forces

        return result

    def predict_energy_and_forces(self, inputs):
        """
        Predict both energy and forces.
        
        Parameters
        ----------
        inputs : dict
            Input dictionary as described in call method
            
        Returns
        -------
        tuple
            (energy, forces) where energy is shape (batch_size,) and forces is shape (batch_size, n_atoms, 3)
        """
        outputs = self(inputs)
        energy = outputs['energy']
        forces = outputs.get('forces', None)
        
        if forces is None and self.predict_forces:
            # Compute forces as negative gradient of energy
            with tf.GradientTape() as tape:
                tape.watch(inputs['R'])
                energy_scalar = tf.reduce_sum(energy)
            forces = -tape.gradient(energy_scalar, inputs['R'])
            # Reshape forces to match expected output format
            forces = tf.reshape(forces, [tf.shape(inputs['batch_seg'])[-1], -1, 3])
        
        return energy, forces


