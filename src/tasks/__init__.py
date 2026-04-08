"""Supervised task builders."""

from tasks.energy_forecast import ForecastTaskConfig, SupervisedDataset, build_supervised_dataset

__all__ = ["ForecastTaskConfig", "SupervisedDataset", "build_supervised_dataset"]
