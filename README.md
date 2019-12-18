# Implementation of HiDDeN in TensorFlow

Based on [_HiDDeN: Hiding data with deep networks (Zhu et al., 2018)_](https://arxiv.org/pdf/1807.09937)

This is a preliminary implementation.

## Installation

Clone the code and install via:

```
python setup.py install
```

## Features

- Noise Layers: identity, gaussian, dropout, crop, cropout, jpeg-mask
- configurable via cfg.py and cmd-arguments


## Tests

```
python -m unittest discover
```

## Open Points

- handle jpeg to yuv transformation
- handle jpeg padding
- test with CIFAR-10
- add summary images
- add logfile summary