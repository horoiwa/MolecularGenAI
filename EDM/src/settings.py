
GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
REMOVE_H = True
MAX_NUM_ATOMS = 30 if not REMOVE_H else 12
MIN_NUM_ATOMS = 3
ATOM_MAP = {"H": 0, "C": 1, "O": 2, "N": 3}
ATOM_MAP_INV = {value: key for key, value in ATOM_MAP.items()}
N_ATOM_TYPES = len(ATOM_MAP)
