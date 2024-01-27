from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

from src import settings, dataset


def remove_mean(x, node_masks):
    n_atoms = tf.reduce_sum(node_masks, axis=1, keepdims=True)       #(B, 1, 1)
    x_mean = (
        tf.reduce_sum(x, axis=1, keepdims=True) / n_atoms
    )
    x_centered = (x - x_mean) * node_masks
    #assert tf.reduce_mean(tf.reduce_sum(x_centered, axis=1)) < 1e-5
    return x_centered


def get_polynomial_noise_schedule(T: int, s=1e-5):
    T = T + 1
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


def sample_gaussian_noise(shape_x, shape_h, node_masks):
    # ガウスノイズをサンプリング後、重心ゼロとなるよう並進移動
    eps_x = remove_mean(
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

        self.T = 1000
        self.scale_x, self.scale_h = 1.0, 4.0
        self.n_layers = n_layers

        self.alphas_cumprod, self.alphas, self.betas = get_polynomial_noise_schedule(self.T)
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

    @tf.function
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
        _x_out = (x - x_in) * node_mask
        x_out = remove_mean(_x_out, node_mask)

        h_out = self.dense_out(h) * node_mask
        h_out = h_out[..., :-1]

        eps = tf.concat([x_out, h_out], axis=-1)

        return eps

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
        x_0 = remove_mean(x, node_masks) / self.scale_x
        h_0 = h / self.scale_h
        z_0 = tf.concat([x_0, h_0], axis=-1)  # (B, N, 3+4)

        # 拡散タイムステップの決定: 0 <= t <= T
        timesteps = tf.random.uniform(
            shape=(x_0.shape[0], 1),
            minval=0,
            maxval=self.T+1,
            dtype=tf.int32
        )
        t = tf.reshape(
            tf.repeat(tf.cast(timesteps / self.T, tf.float32), repeats=N, axis=1),
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

        return loss_z, loss_x, loss_h

    def _sample_tmp(self, x, h, edge_indices, node_masks, edge_masks):
        # 重心が(0,0,0)となるように平行移動し、スケーリング
        B, N = x.shape[0], x.shape[1]
        x_0 = remove_mean(x, node_masks) / self.scale_x
        h_0 = h / self.scale_h
        z_0 = tf.concat([x_0, h_0], axis=-1)  # (B, N, 3+4)

        # 拡散タイムステップの決定: 0 <= t <= T
        timesteps = tf.random.uniform(
            shape=(x_0.shape[0], 1),
            minval=self.T,
            maxval=self.T+1,
            dtype=tf.int32
        )
        t = tf.reshape(
            tf.repeat(tf.cast(timesteps / self.T, tf.float32), repeats=N, axis=1),
            shape=(B, N, 1)
        ) * node_masks

        alphas_cumprod_t = tf.reshape(
            tf.gather(self.alphas_cumprod, indices=timesteps),
            shape=(-1, 1, 1)
        )

        eps = sample_gaussian_noise(shape_x=x_0.shape, shape_h=h_0.shape, node_masks=node_masks)
        z_t = tf.sqrt(alphas_cumprod_t) * z_0 + tf.sqrt(1.0 - alphas_cumprod_t) * eps
        n_atoms = int(tf.reduce_sum(node_masks).numpy())
        mol = self.sample(n_atoms=n_atoms, z_t=z_t)
        return mol


    def sample(self, n_atoms: int, z_t = None, batch_size: int = 1, record_path: Path = None):

        B, N, D = batch_size, settings.MAX_NUM_ATOMS, settings.N_ATOM_TYPES
        node_masks = tf.convert_to_tensor(
            [[[1.] if i < n_atoms else [0.] for i in range(N)]],
            dtype=tf.float32
        )
        _edge_indices, _edge_masks = dataset.get_edges(n_atoms=n_atoms)
        edge_indices = tf.repeat(tf.expand_dims(_edge_indices, axis=0), repeats=B, axis=0)
        edge_masks = tf.repeat(tf.expand_dims(_edge_masks, axis=0), repeats=B, axis=0)

        if z_t is None:
            z_t = sample_gaussian_noise(
                shape_x=(B, N, 3),
                shape_h=(B, N, D),
                node_masks=node_masks
            )

        for timestep in reversed(range(1, self.T+1)):
            _x_t, h_t = z_t[..., :3], z_t[..., 3:]
            x_t = remove_mean(_x_t, node_masks)
            try:
                assert tf.reduce_sum(tf.reduce_sum(x_t, axis=1)) < 1e-5
            except:
                print("Input")
                import pdb; pdb.set_trace()

            x_s, h_s = self.inv_diffusion(x_t, h_t, edge_indices, node_masks, edge_masks, timestep)
            z_s = tf.concat([x_s, h_s], axis=-1)
            z_t = z_s
            if timestep % 10 == 0:
                print(timestep)
                print(z_t[0])

        z_0 = None
        return z_0

    def inv_diffusion(self, x_t, h_t, edge_indices, node_masks, edge_masks, timestep: int):

        B, N, D =  h_t.shape

        timesteps = timestep * tf.ones(shape=(B, 1), dtype=tf.int32)
        alphas_cumprod_t = tf.reshape(
            tf.gather(self.alphas_cumprod, indices=timesteps),
            shape=(-1, 1, 1)
        )
        alphas_cumprod_s = tf.reshape(
            tf.gather(self.alphas_cumprod_prev, indices=timesteps),
            shape=(-1, 1, 1)
        )
        beta_t= tf.reshape(
            tf.gather(self.betas, indices=timesteps),
            shape=(-1, 1, 1)
        )

        t = tf.reshape(
            tf.repeat(tf.cast(timesteps / self.T, tf.float32), repeats=N, axis=1),
            shape=(B, N, 1)
        ) * node_masks

        eps = self(x_t, h_t, t, edge_indices, node_masks, edge_masks)
        eps_x, eps_h = eps[..., :3], eps[..., 3:]

        check = tf.reduce_sum(tf.cast(tf.math.is_nan(eps_x), tf.int32))
        if check > 0:
            print("Nan Assertion")
            import pdb; pdb.set_trace()
        try:
            assert tf.reduce_sum(tf.reduce_sum(eps_x, axis=1)) < 1e-5
        except:
            print("EPS Error")
            import pdb; pdb.set_trace()

        mu_x = (1.0 / tf.sqrt(1.0 - beta_t)) * (eps_x - (beta_t / tf.sqrt(1.0 - alphas_cumprod_t)) * eps_x)
        mu_h = (1.0 / tf.sqrt(1.0 - beta_t)) * (eps_h - (beta_t / tf.sqrt(1.0 - alphas_cumprod_t)) * eps_h)

        variance = beta_t * (1.0 - alphas_cumprod_s) / (1.0 - alphas_cumprod_t)
        sigma = tf.reshape(
            tf.repeat(tf.sqrt(variance), repeats=N, axis=1),
            shape=(B, N, 1)
        ) * node_masks

        noise_x = tf.random.normal(shape=x_t.shape, mean=0., stddev=1.)
        x_s = mu_x + sigma * noise_x

        noise_h = tf.random.normal(shape=h_t.shape, mean=0., stddev=1.)
        h_s = mu_h + sigma * noise_h

        return x_s, h_s


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
            kl.Dense(1, activation=None, use_bias=False, kernel_initializer='truncated_normal'),
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
        x = self.dense_x(feat)
        #x = tf.math.tanh(x) * self.scale_factor
        x = (diff_ij / (d_ij + 1.0)) * x  # (B, N*N, 3) * (B, N*N, 1) -> (B, N*N, 3)
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
