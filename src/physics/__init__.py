"""Double pendulum dynamics, integration, and energy."""

from .energy import compute_energy_timeseries
from .integrator import integrate
from .pendulum import dstate_dt

__all__ = ["compute_energy_timeseries", "dstate_dt", "integrate"]
