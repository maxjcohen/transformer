# Transformers for Time Series

Based on [timeseriestransformer](https://readthedocs.org/projects/timeseriestransformer/badge/?version=latest)

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
