from pathlib import Path
import urllib.request
import functools
import itertools
import tarfile
import random
import io

import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src import settings



def download_qm9(dataset_dir: Path):
    with urllib.request.urlopen(settings.GDB9_URL) as response:
        file = io.BytesIO(response.read())

    with tarfile.open(fileobj=file, mode='r:gz') as tar:
        tar.extractall(path=str(dataset_dir))


def create_tfrecord(dataset_dir: Path, filename: str):
    coords_all, atoms_all, masks_all, edges_all, edge_masks_all = [], [], [], [], []

    sdf_path = dataset_dir / "gdb9.sdf"
    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for n, mol in enumerate(sdf_supplier):
        if mol is None:
            print(f"SKIP {n}: None")
            continue

        n_atoms = mol.GetNumAtoms()
        if n_atoms > settings.MAX_NUM_ATOMS:
            print(f"SKIP {n}: NUM ATOMS {n_atoms} > {settings.MAX_NUM_ATOMS}")
            continue

        mask = [[1.] if i < n_atoms else [0.] for i in range(settings.MAX_NUM_ATOMS)]
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        atoms_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if "F" in atoms_symbol:
            print(f"SKIP {n}: Contains F")
            continue

        atoms_int = [[settings.ATOM_MAP[symbol]] for symbol in atoms_symbol]
        atoms_int += [[0] for _ in range(settings.MAX_NUM_ATOMS - n_atoms)]
        assert len(atoms_int) == settings.MAX_NUM_ATOMS
        atoms_int = tf.convert_to_tensor(atoms_int, dtype=tf.int32)
        atoms_onehot = tf.squeeze(
            tf.one_hot(indices=atoms_int, depth=settings.N_ATOM_TYPES, dtype=tf.float32),
            axis=1,
        ) * mask

        conformer = mol.GetConformer()
        coords = [
            (
                conformer.GetAtomPosition(i).x,
                conformer.GetAtomPosition(i).y,
                conformer.GetAtomPosition(i).z,
            )
            for i in range(n_atoms)
        ]
        coords += [(0., 0., 0.) for _ in range(settings.MAX_NUM_ATOMS - n_atoms)]
        assert len(coords) == settings.MAX_NUM_ATOMS
        coords = tf.convert_to_tensor(coords, dtype=tf.float32)

        coords_all.append(coords)
        atoms_all.append(atoms_onehot)
        masks_all.append(mask)

        edges, edge_masks = get_edges(n_atoms=n_atoms)
        edges_all.append(edges)
        edge_masks_all.append(edge_masks)

    dataset = list(zip(
        coords_all, atoms_all, edges_all, masks_all, edge_masks_all,
        strict=True
    ))
    random.shuffle(dataset)

    filepath = str(dataset_dir / filename)
    with tf.io.TFRecordWriter(filepath) as writer:
        for coords, atoms, edges, masks, edge_masks in tqdm(dataset):
            record = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "coords": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[coords.numpy().tostring()])
                        ),
                        "atoms": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[atoms.numpy().tostring()])
                        ),
                        "edges": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[edges.numpy().tostring()])
                        ),
                        "masks": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[masks.numpy().tostring()])
                        ),
                        "edge_masks": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[edge_masks.numpy().tostring()])
                        ),
                    }
                )
            )
            writer.write(record.SerializeToString())


@functools.cache
def get_edges(n_atoms: int):
    N = settings.MAX_NUM_ATOMS
    indices = list(range(N))
    edges = [(i, j) for i, j in itertools.product(indices, indices)]
    edges = tf.convert_to_tensor(edges, dtype=tf.int32)

    edge_masks = [
        [1] if (i < n_atoms) and (i < n_atoms) and (i != j) else [0]
        for i, j in itertools.product(indices, indices)
    ]
    edge_masks = tf.convert_to_tensor(edge_masks, dtype=tf.float32)
    return edges, edge_masks


def deserialize(serialized_data):
    data = tf.io.parse_example(
        serialized_data,
        features={
            'coords': tf.io.FixedLenFeature([], tf.string),
            'atoms': tf.io.FixedLenFeature([], tf.string),
            'edges': tf.io.FixedLenFeature([], tf.string),
            'masks': tf.io.FixedLenFeature([], tf.string),
            'edge_masks': tf.io.FixedLenFeature([], tf.string),
        }
    )
    coords = tf.reshape(
        tf.io.decode_raw(data['coords'], tf.float32),
        (settings.MAX_NUM_ATOMS, -1)
    )  # (B, N, 3)
    atoms = tf.reshape(
        tf.io.decode_raw(data['atoms'], tf.float32),
        (settings.MAX_NUM_ATOMS, -1)
    )  # (B, N, 4)
    edges = tf.reshape(
        tf.io.decode_raw(data['edges'], tf.int32),
        (-1, 2)
    )  # (B, N*N, 2)
    masks = tf.reshape(
        tf.io.decode_raw(data['masks'], tf.float32),
        (-1, 1)
    )  # (B, N*N, 1)
    edge_masks = tf.reshape(
        tf.io.decode_raw(data['edge_masks'], tf.float32),
        (-1, 1)
    )  # (B, N*N, 1)
    return coords, atoms, edges, masks, edge_masks


def create_dataset_from_tfrecord(tfrecord_path: str, batch_size: int):
    dataset = (
        tf.data.TFRecordDataset(
            filenames=[tfrecord_path],
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        .shuffle(4096, reshuffle_each_iteration=True)
        .repeat()
        .map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
