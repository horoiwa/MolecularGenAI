from pathlib import Path

from src.dataset import download_qm9, create_tfrecord


DATASET_DIR = Path("./data")


def prepare_dataset():
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    if not (DATASET_DIR / "qm9.tfrecor").exists():
        create_tfrecord(dataset_dir=DATASET_DIR)


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    prepare_dataset()
    #train()
    #test()
