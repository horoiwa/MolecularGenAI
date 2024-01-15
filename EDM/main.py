from pathlib import Path
import shutil
import time

import tensorflow as tf

from src.dataset import download_qm9, create_tfrecord, create_dataset_from_tfrecord
from src.models import EquivariantDiffusionModel


DATASET_DIR = Path("./data")
BATCH_SIZE = 64


def load_dataset(filename: str):
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    if not (DATASET_DIR / filename).exists():
        create_tfrecord(dataset_dir=DATASET_DIR, filename=filename)

    dataset = create_dataset_from_tfrecord(
        tfrecord_path=str(DATASET_DIR/filename), batch_size=BATCH_SIZE
    )
    return dataset


def train(checkpoint: int = 0):

    model = EquivariantDiffusionModel()
    dataset = load_dataset(filename="qm9.tfrecord")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-12)

    logdir = Path(__file__).parent / "log"
    savedir = Path(__file__).parent / "checkpoints"
    if checkpoint == 0:
        print("Initilization")
        if logdir.exists():
            shutil.rmtree(logdir)
        if savedir.exists():
            shutil.rmtree(savedir)
    else:
        model.load_weights(f"checkpoints/edm_{checkpoint}")

    summary_writer = tf.summary.create_file_writer(str(logdir))
    now = time.time()
    start = 1 if checkpoint == 0 else checkpoint + 1
    for i, (atom_coords, atom_types, edge_indices, node_masks, edge_masks) in enumerate(dataset, start=start):
        with tf.GradientTape() as tape:
            loss_xh, loss_x, loss_h = model.compute_loss(
                atom_coords, atom_types, edge_indices, node_masks, edge_masks
            )
            loss_xh = tf.reduce_mean(loss_xh)
            loss_x = tf.reduce_mean(loss_x)
            loss_h = tf.reduce_mean(loss_h)

        loss = loss_xh
        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        grads, norm = tf.clip_by_global_norm(grads, 100.0)
        optimizer.apply_gradients(zip(grads, variables))

        if i % 100 == 0:
            elapsed = time.time() - now
            now = time.time()
            tf.print("------------")
            tf.print(i, loss.numpy(), norm.numpy())
            tf.print(f"{elapsed:.1f}sec")
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss_xh, step=i)
                tf.summary.scalar("loss_x", loss_x, step=i)
                tf.summary.scalar("loss_h", loss_h, step=i)
                tf.summary.scalar("global_norm", norm, step=i)

        if i % 10_000 == 0:
            save_path = savedir / f"edm_{i}"
            model.save(str(save_path))

        if i > 1_000_000:
            break


def test(checkpoint: int):
    model = EquivariantDiffusionModel()
    model.load_weights(f"checkpoints/edm_{checkpoint}")
    mol = model.sample(n_atoms=7)


if __name__ == '__main__':
    #train(resume=0)
    #train(checkpoint=280_000)
    test(checkpoint=280_000)
