from pathlib import Path

from src.dataset import download_qm9, create_tfrecord


DATASET_DIR = Path("./data")


def prepare_dataset():
    if not (DATASET_DIR / "gdb9.sdf").exists():
        download_qm9(dataset_dir=DATASET_DIR)

    filename = "QM9.tfrecord"
    if not (DATASET_DIR / filename).exists():
        create_tfrecord(dataset_dir=DATASET_DIR, filename=filename)

    dataset = load_dataset(tfrecord_path=str(DATASET_DIR/filename))


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    prepare_dataset()
    #train()
    #test()
