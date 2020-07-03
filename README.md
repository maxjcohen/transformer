# Transformers for Time Series

[![Documentation Status](https://readthedocs.org/projects/timeseriestransformer/badge/?version=latest)](https://timeseriestransformer.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Latest release](https://img.shields.io/github/release/maxjcohen/transformer.svg)](https://github.com/maxjcohen/transformer/releases/latest)

Implementation of Transformer model (originally from [Attention is All You Need](https://arxiv.org/abs/1706.03762)) applied to Time Series (Powered by [PyTorch](https://pytorch.org/)).

## Transformer model

Transformer are attention based neural networks designed to solve NLP tasks. Their key features are:

- linear complexity in the dimension of the feature vector ;
- paralellisation of computing of a sequence, as opposed to sequential computing ;
- long term memory, as we can look at any input time sequence step directly.

This repo will focus on their application to times series.

## Dataset and application as metamodel

Our use-case is modeling a numerical simulator for building consumption prediction. To this end, we created a dataset by sampling random inputs (building characteristics and usage, weather, ...) and got simulated outputs. We then convert these variables in time series format, and feed it to the transformer.

## Adaptations for time series

In order to perform well on time series, a few adjustments had to be made:

- The embedding layer is replaced by a generic linear layer ;
- Original positional encoding are removed. A "regular" version, better matching the input sequence day/night patterns, can be used instead ;
- A window is applied on the attention map to limit backward attention, and focus on short term patterns.

## Installation

All required packages can be found in `requirements.txt`, and expect to be run with `python3.7`. Note that you may have to install pytorch manually if you are not using pip with a Debian distribution : head on to [PyTorch installation page](https://pytorch.org/get-started/locally/). Here are a few lines to get started with pip and virtualenv:

```bash
$ apt-get install python3.7
$ pip3 install --upgrade --user pip virtualenv
$ virtualenv -p python3.7 .env
$ . .env/bin/activate
(.env) $ pip install -r requirements.txt
```

## Usage

### Downloading the dataset

The dataset is not included in this repo, and must be downloaded manually. It is comprised of two files, `dataset.npz` contains all input and outputs value, `labels.json` is a detailed list of the variables. Please refer to [#2](https://github.com/maxjcohen/transformer/issues/2) for more information.

### Running training script

Using jupyter, run the default `training.ipynb` notebook. All adjustable parameters can be found in the second cell. Careful with the `BATCH_SIZE`, as we are using it to parallelize head and time chunk calculations.

### Outside usage

The `Transformer` class can be used out of the box, see the [docs](https://timeseriestransformer.readthedocs.io/en/latest/?badge=latest) for more info.

```python
from tst import Transformer

net = Transformer(d_input, d_model, d_output, q, v, h, N, TIME_CHUNK, pe)
```

### Building the docs

To build the doc:

```bash
(.env) $ cd docs && make html
```
