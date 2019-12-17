[![Documentation Status](https://readthedocs.org/projects/timeseriestransformer/badge/?version=latest)](https://timeseriestransformer.readthedocs.io/en/latest/?badge=latest) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxjcohen/transformer/demo?filepath=demo_LSTM.ipynb)

Transformers for Time Series
============================

Implementation of Transformer model (originally from [Attention is All You Need](https://arxiv.org/abs/1706.03762)) applied to Time Series (Powered by [PyTorch](https://pytorch.org/)).

## Transformer model
Transformer are attention based neural networks designed to solve NLP tasks. Their key features are:
- linear complexity in the dimension of the feature vector ;
- paralellisation of computing of a sequence, as oposed to sequential computing ;
- long term memory, as we can look at any input time sequence step directly.

This repo will focus on their application to times series.

## Dataset and application as metamodel
Our usecase is modelling a numerical simulator for building consumption prediction. To this end, we created a dataset by sampling random inputs (building caracteristics and usage, meteo, ...) and got simulated outputs. We then convert these variables in time series format, and feed it to the transformer.

## Adaptations for time series
In order to perform well on time series, a few adjustements had to de made:
- Replaced embedding layer for a generic linear layer ;
- Replaced positional encoding with a "regular" version, to better match the input sequence day/night patterns.

## Installation
All required packages can be found in `requirements.txt`, and expect to be run with `python3.7`. If you are not using Debian, pip and virtualenv, you may have to install pytorch manually: remove the last two lines from the `requirements.txt` file and head on [PyTorch website](https://pytorch.org/get-started/locally/).Here are a few lines to get started with pip and virtualenv:

```bash
$ apt-get install python3.7
$ pip3 install --upgrade --user pip virtualenv
$ virtualenv -p python3.7 .env
$ . .env/bin/activate
(.env) $ pip install -r requirements.txt
```

## Usage

### Downloading the dataset
The dataset is not included in this repo, and must be downloaded manually. It is comprised of two files, `dataset.npz` contains all input and outputs value, `labels.json` is a detailled list of the variables.

### Running training script
Using jupyter, run the default `training.ipynb` notebook. All adjustable parameters can be found in the second cell. Carefull with the `BATCH_SIZE`, as we are using it to parallelize head and time chunk calculations. 

### Outside usage
The `Transformer` class can be used out of the box, [docs](https://timeseriestransformer.readthedocs.io/en/latest/Transformer.html) for more infos.

```python
from src.Transformer import Transformer

net = Transformer(d_input, d_model, d_output, q, v, h, K, N, TIME_CHUNK, pe)
```

### Buidling the docs
To build the doc:
```bash
(.env) $ cd docs && make html
```