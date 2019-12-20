# Implementation of HiDDeN in TensorFlow

Based on [_HiDDeN: Hiding data with deep networks (Zhu et al., 2018)_](https://arxiv.org/pdf/1807.09937)

This is a preliminary implementation.

## Installation

Clone the code and install via:

```
pip install -r requirements.txt
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
- test with CIFAR-10
- add logfile summary
- create matplotlib figures
- handle multi noise, randomly choose
- use 'real' jpeg for testing
- add parameters for noise layers
- test COCO on AWS
- implement from image dir, with crop option
- add peak signal to noise ratio metric between cover and encoded image, ensure correctness for jpeg and yuv
- COCO batch size is 12
- fix crop layer: dont padd with zeroes - verify 
- verify random seed works as expected
- investigate low distortion loss while tensorboard shows visual dissimilarity
- manually investigate input on sample image
- implement logloss for summaries?
- log flags / configuration
