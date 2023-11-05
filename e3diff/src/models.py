import tensorflow as tf


def align_with_centroid(coords, masks):
    n_atoms = tf.reduce_sum(masks, axis=1, keepdims=True)       #(B, 1)
    coords_mean = (
        tf.reduce_sum(coords, axis=1, keepdims=True) / n_atoms
    )
    coords = (coords - coords_mean) * masks
    #assert tf.reduce_sum(tf.reduce_mean(coords, axis=1)) < 1e-5
    return coords


class EquivariantDiffusionModel(tf.keras.Model):

    def __init__(self):
        super(EquivariantDiffusionModel, self).__init__()
        self.diffusion_steps = 1000

        self.scale_coord, self.scale_atom = 1.0, 4.0

    def call(self, x):
        """ predict added noise ε
        """
        eps = x
        return eps

    def forward_diffusion(self, x_0, timesteps):
        pass

    def inv_diffusion(self, x_t, timesteps):
        pass

    def sample(self, batch_size: int):
        pass

    def compute_loss(
            self, coords, atoms,
            edges, masks, edge_masks,
        ):
        coords = align_with_centroid(coords, masks)
        coords, atoms = self.normalize(coords, atoms)

        loss = None
        return loss

    def normalize(self, coords, atoms):
        coords = coords / self.scale_coord
        atoms = atoms / self.scale_atom
        return coords, atoms

    def inv_normalize(self, coords, atoms):
        coords = coords * self.scale_coord
        atoms = atoms * self.scale_atom
        return coords, atoms

