import tensorflow as tf
import numpy as np

from src import settings


def align_with_center_of_gravity(coords, masks):
    n_atoms = tf.reduce_sum(masks, axis=1, keepdims=True)       #(B, 1)
    coords_mean = (
        tf.reduce_sum(coords, axis=1, keepdims=True) / n_atoms
    )
    coords_centered = (coords - coords_mean) * masks
    try:
        assert tf.reduce_sum(tf.reduce_mean(coords_centered, axis=1)) < 1e-5
    except:
        import pdb; pdb.set_trace()
    return coords_centered

def get_noise_schedule(T: int, s=0.008):
    """cosine_schedule"""
    t = tf.cast(tf.range(0, T+1), tf.float32)
    _alphas_cumprod = tf.cos(0.5 * np.pi * ((t / T) + s ) / (1 + s))**2
    alphas_cumprod = _alphas_cumprod / _alphas_cumprod[0]
    _betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = tf.clip_by_value(_betas, clip_value_min=0., clip_value_max=0.999)
    alphas = 1. - betas
    return alphas, betas

def sample_gaussian_noise(shape_x, shape_h, masks):
    # ガウスノイズをサンプリング後、重心ゼロとなるよう並進移動
    eps_x = align_with_center_of_gravity(
        tf.random.normal(shape=shape_x, mean=0., stddev=1.) * masks,
        masks,
    )
    eps_h = tf.random.normal(shape=(shape_h), mean=0., stddev=1.) * masks
    eps = tf.concat([eps_x, eps_h], axis=-1)
    return eps


class EquivariantDiffusionModel(tf.keras.Model):

    def __init__(self):
        super(EquivariantDiffusionModel, self).__init__()
        self.num_steps = 1000
        self.scale_coord, self.scale_atom = 1.0, 4.0

        # α: √1-β , σ: √β より alphas=√1-σ^2
        self.alphas, self.betas = get_noise_schedule(T=self.num_steps)
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.alphas_cumprod_prev = tf.concat([[1.], self.alphas_cumprod[:-1]], axis=0)
        self.variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def call(self, x):
        """ predict added noise ε
        """
        eps = x
        return eps

    def forward_diffusion(self, x_0, timesteps):
        pass

    def inv_diffusion(self, x_t, timesteps):
        pass

    def compute_loss(
            self, coords, atoms,
            edges, masks, edge_masks,
        ):
        # 重心が(0,0,0)となるように平行移動
        coords = align_with_center_of_gravity(coords, masks)

        # 入力値のスケーリング
        x_0, h_0 = self.normalize(coords, atoms)
        z_0 = tf.concat([x_0, h_0], axis=-1)  # (B, N, 3+4)

        # 拡散タイムステップの決定: 0 <= t <= T
        t = tf.random.uniform(
            shape=(x_0.shape[0], 1), minval=0, maxval=self.num_steps+1, dtype=tf.int32
        )

        alphas_t = 1.
        sigmas_t = 1.

        # ガウスノイズのサンプリング
        eps = sample_gaussian_noise(shape_x=x_0.shape, shape_h=h_0.shape, masks=masks)

        z_t = alphas_t * z_0 + sigmas_t * eps
        x_t, h_t = z_t[..., :3], z_t[..., 3:]

        eps_pred = self(x_t, h_t, t)

        is_t0 = tf.cast(t == 0, tf.int32)
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


class EquivariantGCNN(tf.keras.Model):

    def __init__(self):
        super(EquivariantGCNN, self).__init__()

    def call(self, x):
        return x
