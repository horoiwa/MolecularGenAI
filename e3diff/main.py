from pathlib import Path

import tensorflow as tf

from src.dataset import download_qm9, create_tfrecord, create_dataset_from_tfrecord
from src.models import EquivariantDiffusionModel


DATASET_DIR = Path("./data")
BATCH_SIZE = B = 10
#BATCH_SIZE = B = 64


def load_dataset(filename: str):
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    if not (DATASET_DIR / filename).exists():
        create_tfrecord(dataset_dir=DATASET_DIR, filename=filename)

    dataset = create_dataset_from_tfrecord(
        tfrecord_path=str(DATASET_DIR/filename), batch_size=BATCH_SIZE
    )

    #for (coords, atoms, edges, masks, edge_masks) in dataset:
    #    indices_from, indices_to = edges[..., 0:1], edges[..., 1:2]
    #    nodes_from = tf.gather_nd(atoms, indices_from, batch_dims=1)
    #    nodes_to = tf.gather_nd(atoms, indices_to, batch_dims=1)
    #    break

    return dataset


def train():
    dataset = load_dataset(filename="qm9.tfrecord")
    model = EquivariantDiffusionModel()
    optimizer = tf.keras.optimizers.AdamW(lr=1e-4, weight_decay=1e-12)

    for (atom_coords, atom_types, edges, masks, edge_masks) in dataset:
        #with tf.GradientTape() as tape:
        loss = model.compute_loss(
            atom_coords, atom_types, edges, masks, edge_masks
        )

        #variables = model.network.trainable_variables
        #grads = tape.garadient(loss, variables)
        #optimizer.apply_gradients(zip(grads, variables))




def test():
    pass


if __name__ == '__main__':
    train()
    test()
