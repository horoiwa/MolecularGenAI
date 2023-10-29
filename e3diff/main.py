from pathlib import Path


DATASET_DIR = Path(__file__).parents[0] / "data"


def load_dataset():
    import pdb; pdb.set_trace()
    pass


def train():
    dataset = load_dataset()
    pass


def test():
    pass


if __name__ == '__main__':
    load_dataset()
    train()
    test()
