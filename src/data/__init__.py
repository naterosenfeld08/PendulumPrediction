"""Trajectory schema and IO utilities."""

from data.generation import GenerationConfig, generate_trajectory_corpus
from data.io import load_record, save_record
from data.schema import REQUIRED_ENERGY_CHANNELS, TrajectoryRecord, validate_record
from data.splits import split_ids

__all__ = [
    "REQUIRED_ENERGY_CHANNELS",
    "TrajectoryRecord",
    "GenerationConfig",
    "generate_trajectory_corpus",
    "load_record",
    "save_record",
    "split_ids",
    "validate_record",
]
