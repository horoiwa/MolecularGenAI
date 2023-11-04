from pathlib import Path
import urllib.request
import tarfile
import io

import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors

from src import settings



def download_qm9(dataset_dir: Path):
    with urllib.request.urlopen(settings.GDB9_URL) as response:
        file = io.BytesIO(response.read())

    with tarfile.open(fileobj=file, mode='r:gz') as tar:
        tar.extractall(path=str(dataset_dir))


def create_tfrecord(dataset_dir: Path):

    X_coords, X_atoms, X_masks = [], [], []

    sdf_path = dataset_dir / "gdb9.sdf"
    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for n, mol in enumerate(sdf_supplier):
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
            tf.one_hot(indices=atoms_int, depth=len(settings.ATOM_MAP)+1, dtype=tf.float32),
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

        X_coords.append(coords)
        X_atoms.append(atoms_onehot)
        X_masks.append(mask)

        import pdb; pdb.set_trace()

