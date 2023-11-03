from pathlib import Path
import urllib.request
import tarfile
import io

from rdkit import Chem

GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"


def download_qm9(dataset_dir: Path):
    with urllib.request.urlopen(GDB9_URL) as response:
        file = io.BytesIO(response.read())

    with tarfile.open(fileobj=file, mode='r:gz') as tar:
        tar.extractall(path=str(dataset_dir))


def create_tfrecord(dataset_dir: Path):
    sdf_path = dataset_dir / "gdb9.sdf"
    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in sdf_supplier:
        n_atoms = mol.GetNumAtoms()

