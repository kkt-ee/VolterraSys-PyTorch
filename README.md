# VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases

[![PyPI Version](https://img.shields.io/pypi/v/VolterraSys-PyTorch?label=PyPI&color=gold)](https://pypi.org/project/VolterraSys-PyTorch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/VolterraSys-PyTorch)](https://pypi.org/project/VolterraSys-PyTorch/)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-deepgreen.svg?style=flat)](https://github.com/kkt-ee/VolterraSys-PyTorch/blob/master/LICENSE)

`VolterraSys-PyTorch` provides PyTorch modules for trainable 1D Volterra kernels of general order in wavelet and natural bases. Tensors use channels-last layout: `(batch, length, channels)`.

<!-- Copyright 2026 Kishore Kumar Tarafdar.  
Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE). -->

## Capabilities

- Linear shift-invariant 1D kernels: `LSIVolterra1D`.
- Quadratic shift-invariant 1D kernels: `QSIVolterra1D`.
- Shift-variant 1D kernels: `LinearVolterra1D` and `Volterra1D(m=...)` for general order.
- Orthogonal and biorthogonal wavelet-domain computation through `fdwt`.
- Natural-domain computation with `wave=None`.
- Trainable kernels with PyTorch autograd, optimizers, and `state_dict` serialization.

## Installation

```bash
pip install VolterraSys-PyTorch
```

## Minimal example

```python
import torch

from volterrasys.LSIVolterra1D import LSIVolterra1D
from volterrasys.QSIVolterra1D import QSIVolterra1D
from volterrasys.LinearVolterra1D import LinearVolterra1D
from volterrasys.Volterra1D import Volterra1D

x = torch.randn(2, 32, 1)

lsi = LSIVolterra1D(filters=2, kernel_size=4, wave="bior1.3")
qsi = QSIVolterra1D(filters=2, kernel_size=4, wave="haar")
linear = LinearVolterra1D(filters=2, Ny=16, wave=None)
general = Volterra1D(m=3, filters=2, Ny=16, wave="bior1.3")

y_lsi = lsi(x)       # (2, 32, 2)
y_qsi = qsi(x)       # (2, 32, 2)
y_linear = linear(x) # (2, 16, 2)
y_general = general(x) # (2, 16, 2)
```

The wavelet-domain and natural-domain implementations use the same channels-last tensor layout. `filters` is the number of output channels.

## TensorFlow provenance

This repository is an operation-by-operation PyTorch port of [kkt-ee/VolterraSys](https://github.com/kkt-ee/VolterraSys). The initial port was based on TensorFlow commit `4ef3877` from `bugfix/biortho-linear1d`, marked by the `tf-final-before-pytorch-port` tag. The original TensorFlow Git history is retained as this repository's ancestry.

The port preserves the TensorFlow kernel construction, circular indexing, wavelet-basis transformations, and `einsum` contractions. The corresponding PyTorch dependency is [`fdwt`](https://pypi.org/project/fdwt/).

## Citation

If this package proves useful in related work, please cite the following thesis, whose Chapter 2 presents the underlying theory and computational details:

```bibtex
@misc{tarafdar2026interpretablefrugallearningsystems,
      title={Interpretable and Frugal Learning Systems Employing Multiresolution Pyramids and Volterra Kernels},
      author={Kishore Kumar Tarafdar},
      year={2026},
      eprint={2606.15011},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.15011},
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

* * *

***VolterraSys-PyTorch (C) 2026 Kishore Kumar Tarafdar, भारत*** 🇮🇳