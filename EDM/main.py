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

    return dataset


def train():
    dataset = load_dataset(filename="qm9.tfrecord")
    model = EquivariantDiffusionModel()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-12)

    for (atom_coords, atom_types, edge_indices, node_masks, edge_masks) in dataset:
        #with tf.GradientTape() as tape:
        loss = model.compute_loss(
            atom_coords, atom_types, edge_indices, node_masks, edge_masks
        )

        #variables = model.network.trainable_variables
        #grads = tape.garadient(loss, variables)
        #optimizer.apply_gradients(zip(grads, variables))




def test():
    pass


if __name__ == '__main__':
    train()
    test()
