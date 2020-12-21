# Transformers for Time Series
[![PyPI version](https://badge.fury.io/py/time-series-transformer.svg)](https://badge.fury.io/py/time-series-transformer) [![travis](https://travis-ci.org/DanielAtKrypton/time_series_transformer.svg?branch=master)](https://travis-ci.org/github/DanielAtKrypton/time_series_transformer) [![codecov](https://codecov.io/gh/DanielAtKrypton/time_series_transformer/branch/master/graph/badge.svg)](https://codecov.io/gh/DanielAtKrypton/time_series_transformer) [![GitHub license](https://img.shields.io/github/license/DanielAtKrypton/time_series_transformer)](https://github.com/DanielAtKrypton/time_series_transformer) [![Requirements Status](https://requires.io/github/DanielAtKrypton/time_series_transformer/requirements.svg?branch=master)](https://requires.io/github/DanielAtKrypton/time_series_transformer/requirements/?branch=master)

Based on [timeseriestransformer](https://readthedocs.org/projects/timeseriestransformer/badge/?version=latest).

## Installation

```terminal
.\scripts\init-env.ps1
```

## Usage

```python
from flights_time_series_dataset import FlightsDataset
from time_series_predictor import TimeSeriesPredictor
from time_series_transformer import Transformer

tsp = TimeSeriesPredictor(
    Transformer(),
    max_epochs=50,
    train_split=None,
)

tsp.fit(FlightsDataset())
```

### Test

To test the package simply run the following command from project's root folder.

```bash
pytest -s
```
