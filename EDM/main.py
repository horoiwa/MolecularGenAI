from pathlib import Path
import shutil
import time

import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDetermineBonds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

from src.dataset import download_qm9, create_tfrecord, create_dataset_from_tfrecord
from src.models import EquivariantDiffusionModel
from src import settings


DATASET_DIR = Path("./data")
BATCH_SIZE = 48


def load_dataset(filename: str, batch_size: int = BATCH_SIZE):
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    if not (DATASET_DIR / filename).exists():
        create_tfrecord(dataset_dir=DATASET_DIR, filename=filename)

    dataset = create_dataset_from_tfrecord(
        tfrecord_path=str(DATASET_DIR/filename), batch_size=batch_size
    )
    return dataset


def train(checkpoint: int = 0):

    model = EquivariantDiffusionModel()
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

    n = 0
    i = 1 if checkpoint == 0 else checkpoint + 1
    while True:
        print("=========")
        print(f"Epoch: {n}")
        print("=========")

        dataset = load_dataset(filename="qm9.tfrecord")
        for (atom_coords, atom_types, edge_indices, node_masks, edge_masks) in dataset:
            with tf.GradientTape() as tape:
                loss_xh, loss_x, loss_h = model.compute_loss(
                    atom_coords, atom_types, edge_indices, node_masks, edge_masks
                )
                loss_xh = tf.reduce_mean(tf.reduce_sum(loss_xh, axis=1))
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
            i += 1

        n += 1


def create_xyz(coords: np.ndarray, atom_types: np.ndarray, n_atoms: int):
    coords = coords.numpy()
    atom_types = atom_types.numpy()

    xyz_block = f"{n_atoms}\n"
    xyz_block += "\n"
    for i in range(n_atoms):
        x, y, z = coords[i].tolist()
        n = np.argmax(atom_types[i])
        atom = settings.ATOM_MAP_INV[n]
        xyz_block += f"{atom}      {x:.6f}   {y:.6f}   {z:.6f}\n"
    mol = Chem.MolFromXYZBlock(xyz_block)
    if settings.REMOVE_H:
        mol = Chem.AddHs(mol)
    try:
        rdDetermineBonds.DetermineBonds(mol)
    except:
        pass
    return mol


def create_animation(history, n_atoms, save_path):

    # 3Dプロットの初期設定
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    fig = plt.figure(figsize=(10, 5))
    #ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(121, projection='3d')
    ax3 = fig.add_subplot(122, projection='3d')
    cmap = {"H": "white", "C": "gray", "O": "red", "N": "blue"}
    smap = {"H": 50, "C": 150, "O": 150, "N": 150}
    emap = {"H": "black", "C": "black", "O": "black", "N": "black"}

    # アニメーションのための更新関数
    def update(frame: int):
        for key, ax in {"current": ax2, "end": ax3}.items():
            ax.clear()  # 前のフレームのプロットをクリア
            ax.view_init(elev=30, azim=-65)
            if key == "current":
                timestep, coords, h = history[frame]
                print(timestep)
            elif key == "start":
                timestep, coords, h = history[0]
            elif key == "end":
                timestep, coords, h = history[-1]

            coords, h = coords.numpy(), h.numpy()
            for i in range(n_atoms):
                x, y, z = coords[i].tolist()
                atom_type = settings.ATOM_MAP_INV[np.argmax(h[i])]
                ax.scatter(
                    x, y, z, color=cmap[atom_type], s=smap[atom_type],
                    edgecolors=emap[atom_type], linewidths=0.4
                )
                ax.set_title(f"timestep: {timestep}")

            ax.set_xlim(-3, 3)  # z軸の範囲を固定
            ax.set_ylim(-3, 3)  # z軸の範囲を固定
            ax.set_zlim(-3, 3)  # z軸の範囲を固定
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    # アニメーションの作成
    ani = FuncAnimation(
        fig, update, frames=np.arange(0, len(history)),
        interval=100, repeat_delay=3000
    )
    writer = PillowWriter(fps=15)
    ani.save(save_path, writer=writer)


def write_to_sdf(mol, save_path):
    with Chem.SDWriter(save_path) as writer:
        writer.write(mol)


def generate(checkpoint: int, n_atoms: int):

    model = EquivariantDiffusionModel()
    model.load_weights(f"checkpoints/edm_{checkpoint}")

    x_0, h_0, history = model.sample(n_atoms=n_atoms)
    create_animation(history, n_atoms, save_path="out/trajectory.gif")
    mol = create_xyz(x_0, h_0, n_atoms=n_atoms)
    write_to_sdf(mol, save_path="out/sampled.sdf")



if __name__ == '__main__':
    #train(checkpoint=350_000)
    generate(checkpoint=350_000, n_atoms=20)
