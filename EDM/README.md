Minimal TF2 implementation of (Equivariant Diffusion for Molecule Generation in 3D)[https://arxiv.org/abs/2203.17003]


official implementation (pytorch):
https://github.com/ehoogeboom/e3_diffusion_for_molecules


## Requirements

```
python==3.11.5
tensorflow==2.13.1
rdkit-2023.9.1
```


tf.Tensor(
[[[ 1.8732  1.3887  0.1125]
  [ 1.1628  0.0604  0.1473]
  [ 1.784  -1.0088  0.8773]
  [ 0.4998 -0.5248  1.328 ]
  [ 0.4802  0.1515  2.6795]
  [-0.6104 -1.5385  1.0913]
  [-0.4179 -2.7227  1.9107]
  [-1.3386 -3.1773  2.8012]
  [-2.4221 -2.6727  3.0097]
  [ 0.      0.      0.    ]
  [ 0.      0.      0.    ]
  [ 0.      0.      0.    ]]], shape=(1, 12, 3), dtype=float32)
tf.Tensor(
[[[0. 1. 0. 0.]
  [0. 1. 0. 0.]
  [0. 0. 1. 0.]
  [0. 1. 0. 0.]
  [0. 1. 0. 0.]
  [0. 1. 0. 0.]
  [0. 0. 0. 1.]
  [0. 1. 0. 0.]
  [0. 0. 1. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]], shape=(1, 12, 4), dtype=float32)
