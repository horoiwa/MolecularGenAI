from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

from src import settings


def align_with_center_of_gravity(x, node_masks):
    n_atoms = tf.reduce_sum(node_masks, axis=1, keepdims=True)       #(B, 1, 1)
    x_mean = (
        tf.reduce_sum(x, axis=1, keepdims=True) / n_atoms
    )
    x_centered = (x - x_mean) * node_masks
    #assert tf.reduce_mean(tf.reduce_sum(x_centered, axis=1)) < 1e-5
    return x_centered


def get_polynomial_noise_schedule(num_steps: int, s=1e-5):
    T = num_steps + 1
    t = tf.cast(tf.linspace(0, T, T), tf.float32)

    alphas_cumprod = (1.0 - (t/(T)) ** 2) ** 2

    # clipping procedure (Appendix B)
    _alphas_cumprod = tf.concat([[1.], alphas_cumprod], axis=-1)
    alphas = tf.clip_by_value(
        _alphas_cumprod[1:] / _alphas_cumprod[:-1],
        clip_value_max=1.0,
        clip_value_min=0.001
    )
    betas = 1.0 - alphas
    alphas_cumprod = (1 - 2 * s) * tf.math.cumprod(alphas) + s

    return alphas_cumprod, alphas, betas


def get_cosine_noise_schedule(T: int, s=0.008):
    """cosine_schedule"""
    t = tf.cast(tf.range(0, T+1), tf.float32)
    _alphas_cumprod = tf.cos(0.5 * np.pi * ((t / T) + s ) / (1 + s))**2
    alphas_cumprod = _alphas_cumprod / _alphas_cumprod[0]
    _betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = tf.clip_by_value(_betas, clip_value_min=0., clip_value_max=0.999)
    alphas = 1. - betas
    return alphas, betas


def sample_gaussian_noise(shape_x, shape_h, node_masks):
    # ガウスノイズをサンプリング後、重心ゼロとなるよう並進移動
    eps_x = align_with_center_of_gravity(
        tf.random.normal(shape=shape_x, mean=0., stddev=1.) * node_masks,
        node_masks,
    )
    eps_h = tf.random.normal(shape=(shape_h), mean=0., stddev=1.) * node_masks
    eps = tf.concat([eps_x, eps_h], axis=-1)
    return eps


def compute_distance(x, edge_indices):
    indices_from, indices_to = edge_indices[..., 0:1], edge_indices[..., 1:2]
    x_i = tf.gather_nd(x, indices_from, batch_dims=1)
    x_j = tf.gather_nd(x, indices_to, batch_dims=1)
    d_ij = tf.sqrt(tf.reduce_sum((x_i- x_j)**2, axis=-1, keepdims=True))
    return d_ij


class EquivariantDiffusionModel(tf.keras.Model):

    def __init__(self, n_layers: int = 9):
        super(EquivariantDiffusionModel, self).__init__()

        self.num_steps = 1000
        self.scale_x, self.scale_h = 1.0, 4.0
        self.n_layers = n_layers

        self.alphas_cumprod, self.alphas, self.betas = get_polynomial_noise_schedule(self.num_steps)
        self.alphas_cumprod_prev = tf.concat([[1.], self.alphas_cumprod[:-1]], axis=0)

        self.dense_in = kl.Dense(256, activation=None, kernel_initializer='truncated_normal')
        self.egnn_blocks = [EquivariantGNNBlock() for _ in range(self.n_layers)]
        self.dense_out = kl.Dense(
            len(settings.ATOM_MAP) + 1, activation=None, kernel_initializer='truncated_normal'
        )

    def save(self, save_path: str):
        self.save_weights(save_path)

    def load(self, load_path: str):
        self.load_weights(load_path)

    def call(self, x_in, h_in, t, edge_indices, node_mask, edge_mask):
        """ predict noise ε_t
        """
        d_ij_in = compute_distance(x_in, edge_indices) * edge_mask
        x, h = x_in, tf.concat([h_in, t], axis=-1)
        h = self.dense_in(h)
        for egnn_block in self.egnn_blocks:
            x, h = egnn_block(
                x=x, h=h, edge_attr=d_ij_in, edge_indices=edge_indices,
                node_mask=node_mask, edge_mask=edge_mask,
            )
        x = (x - x_in) * node_mask
        x_out = align_with_center_of_gravity(x, node_mask)

        h_out = self.dense_out(h) * node_mask
        h_out = h_out[..., :-1]

        eps = tf.concat([x_out, h_out], axis=-1)

        return eps

    #@tf.function
    def compute_loss(self, x, h, edge_indices, node_masks, edge_masks):
        """
        Notes:
            B: バッチサイズ
            N: 最大原子数(settings.MAX_NUM_ATOMS)
        Args:
            x: xyz座標, shape==(B, N, 3)
            h: OneHot encoded原子タイプ, shape==(B, N, len(settings.ATOM_MAP))
            edge_indices: すべての２つの原子の組み合わせ番号 shape==(B, N*N, ...)
            node_masks: paddingされたダミー原子でないか, shape==(B, N, ...)
            edge_masks: エッジの両端がダミー原子でないか, shape==(B, N*N, ...)
        """

        # 重心が(0,0,0)となるように平行移動し、スケーリング
        B, N = x.shape[0], x.shape[1]
        x_0 = align_with_center_of_gravity(x, node_masks) / self.scale_x
        h_0 = h / self.scale_h
        z_0 = tf.concat([x_0, h_0], axis=-1)  # (B, N, 3+4)

        # 拡散タイムステップの決定: 0 <= t <= T
        timesteps = tf.random.uniform(
            shape=(x_0.shape[0], 1),
            #minval=0,
            minval=self.num_steps,
            maxval=self.num_steps+1,
            dtype=tf.int32
        )
        t = tf.reshape(
            tf.repeat(tf.cast(timesteps / self.num_steps, tf.float32), repeats=N, axis=1),
            shape=(B, N, 1)
        ) * node_masks
        alphas_cumprod_t = tf.reshape(
            tf.gather(self.alphas_cumprod, indices=timesteps),
            shape=(-1, 1, 1)
        )

        # 順拡散プロセス
        eps = sample_gaussian_noise(shape_x=x_0.shape, shape_h=h_0.shape, node_masks=node_masks)
        z_t = tf.sqrt(alphas_cumprod_t) * z_0 + tf.sqrt(1.0 - alphas_cumprod_t) * eps
        x_t, h_t = z_t[..., :3], z_t[..., 3:]

        # E3同変GNNによるノイズ予測とL2ロス算出
        eps_pred = self(x_t, h_t, t, edge_indices, node_masks, edge_masks)

        loss_z = 0.5 * (eps - eps_pred) **2
        loss_x, loss_h = loss_z[..., :3], loss_z[..., 3:]
        import pdb; pdb.set_trace()
        # if debug := True:
        #     eps_x, eps_h = eps[..., :3], eps[..., 3:]
        #     z_0_x, z_0_h = z_0[..., :3], z_0[..., 3:]
        #     z_t_x, z_t_h = z_t[..., :3], z_t[..., 3:]
        #     eps_x_p, eps_h_p = eps_pred[..., :3], eps_pred[..., 3:]
        #     import pdb; pdb.set_trace()
        #     x_0_pred = (1.0 / tf.sqrt(alphas_cumprod_t)) * x_t - (tf.sqrt((1.0 - alphas_cumprod_t) / alphas_cumprod_t)) * eps_pred[..., :3]
        #     h_0_pred = (1.0 / tf.sqrt(alphas_cumprod_t)) * h_t - (tf.sqrt((1.0 - alphas_cumprod_t) / alphas_cumprod_t)) * eps_pred[..., 3:]
        #     loss_x_all = 0.5 * (eps[..., :3] - eps_pred[..., :3]) **2
        #     loss_h_all = 0.5 * (eps[..., 3:] - eps_pred[..., 3:]) **2
        #     loss_x = tf.reduce_mean(loss_x_all)
        #     loss_h = tf.reduce_mean(loss_h_all)

        return loss_z, loss_x, loss_h

    def sample_molecule(self, x):
        pass

    def inv_diffusion(self, x_t, timesteps):
        pass



class EquivariantGNNBlock(tf.keras.Model):

    def __init__(self):
        super(EquivariantGNNBlock, self).__init__()
        self.dense_e = tf.keras.Sequential([
            kl.Dense(256, activation=tf.nn.silu, kernel_initializer='truncated_normal'),
            kl.Dense(256, activation=tf.nn.silu, kernel_initializer='truncated_normal'),
        ])
        self.e_attention = kl.Dense(1, activation='sigmoid', kernel_initializer='truncated_normal')

        self.dense_h = tf.keras.Sequential([
            kl.Dense(256, activation=tf.nn.silu, kernel_initializer='truncated_normal'),
            kl.Dense(256, activation=None, kernel_initializer='truncated_normal'),
        ])
        self.dense_x = tf.keras.Sequential([
            kl.Dense(256, activation=tf.nn.silu, kernel_initializer='truncated_normal'),
            kl.Dense(256, activation=tf.nn.silu, kernel_initializer='truncated_normal'),
            kl.Dense(1, activation="tanh", use_bias=False, kernel_initializer='glorot_uniform'),
        ])
        self.scale_factor = 10.0


    def call(self, x, h, edge_attr, edge_indices, node_mask, edge_mask):
        indices_i, indices_j = edge_indices[..., 0:1], edge_indices[..., 1:2]

        x_i = tf.gather_nd(x, indices_i, batch_dims=1)
        x_j = tf.gather_nd(x, indices_j, batch_dims=1)

        diff_ij = (x_i - x_j) * edge_mask
        d_ij = tf.sqrt(tf.reduce_sum(diff_ij**2, axis=-1, keepdims=True))

        h_i = tf.gather_nd(h, indices_i, batch_dims=1)
        h_j = tf.gather_nd(h, indices_j, batch_dims=1)

        feat = tf.concat([h_i, h_j, d_ij**2, edge_attr], axis=-1) * edge_mask
        x_out = self.update_x(x, diff_ij, d_ij, feat, indices_i) * node_mask
        h_out = self.update_h(h, feat, indices_i) * node_mask
        return x_out, h_out

    def update_h(self, h_in, feat, indices_i):
        m_ij = self.dense_e(feat)
        e_ij = self.e_attention(m_ij)
        em_ij = e_ij * m_ij
        em_agg = segmnt_sum_by_node(em_ij, indices_i)
        h_out = h_in + self.dense_h(tf.concat([h_in, em_agg], axis=-1))
        return h_out

    def update_x(self, x_in, diff_ij, d_ij, feat, indices_i):
        # tanhでのアクティベーション後にリスケール
        x = diff_ij / (d_ij + 1.0) * self.dense_x(feat) * self.scale_factor  # (B, N*N, 3) * (B, N*N, 1) -> (B, N*N, 3)
        x_agg = segmnt_sum_by_node(x, indices_i)
        x_out = x_in + x_agg
        return x_out


def segmnt_sum_by_node(data, indices_i):
    """
    やりたいことは単なるdata.groupby(indices_i).sum()だが、
    tf.math.unsorted_segment_sumの仕様上、迂遠な処理が必要
    """
    B, NN, D = data.shape
    data = tf.reshape(data, shape=(B*NN, D))  # (B, NN, D) -> (B*NN, D)
    indices = tf.reshape(
        tf.reshape(tf.range(B), shape=(B, 1)) * settings.MAX_NUM_ATOMS + tf.squeeze(indices_i, axis=-1),
        shape=(B*NN,),
    )
    num_segments = B * settings.MAX_NUM_ATOMS
    agg = tf.reshape(
        tf.math.unsorted_segment_sum(data=data, segment_ids=indices, num_segments=num_segments),
        shape=(B, -1, D)
    )
    return agg
