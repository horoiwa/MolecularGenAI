from pathlib import Path
import shutil
import time

import tensorflow as tf

from src.dataset import download_qm9, create_tfrecord, create_dataset_from_tfrecord
from src.models import EquivariantDiffusionModel


DATASET_DIR = Path("./data")
BATCH_SIZE = 48


def load_dataset(filename: str):
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    if not (DATASET_DIR / filename).exists():
        create_tfrecord(dataset_dir=DATASET_DIR, filename=filename)

    dataset = create_dataset_from_tfrecord(
        tfrecord_path=str(DATASET_DIR/filename), batch_size=BATCH_SIZE
    )

    return dataset


def train(resume: int = 0):

    model = EquivariantDiffusionModel()
    dataset = load_dataset(filename="qm9.tfrecord")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-12)

    if resume == 0:
        logdir = Path(__file__).parent / "log"
        if logdir.exists():
            shutil.rmtree(logdir)
        summary_writer = tf.summary.create_file_writer(str(logdir))

        savedir = Path(__file__).parent / "checkpoints"
        if savedir.exists():
            shutil.rmtree(savedir)
    else:
        model.load_weights("checkpoints/edm")

    now = time.time()
    start = 1 if resume == 0 else resume
    for i, (atom_coords, atom_types, edge_indices, node_masks, edge_masks) in enumerate(dataset, start=start):

        with tf.GradientTape() as tape:
            loss = model.compute_loss(
                atom_coords, atom_types, edge_indices, node_masks, edge_masks
            )

        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if i % 100 == 0:
            elapsed = time.time() - now
            now = time.time()
            tf.print("------------")
            tf.print(i, loss.numpy())
            tf.print(f"{elapsed:.1f}")
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=i)

        if i % 50_000 == 0:
            save_path = savedir / "edm"
            model.save(str(save_path))


def test():
    pass


if __name__ == '__main__':
    train()
    test()
