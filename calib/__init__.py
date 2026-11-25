"""Fictitious health-score calibration data with buried true weights.

Public entry points
-------------------
``calib.generate_synthetic_health.generate``  run the whole pipeline in memory
``calib.generate_synthetic_health.main``      command line entry point
``calib.config.load_config``                  load and validate a YAML config
``calib.schema``                              column contract of both tables

Command line::

    python -m calib.generate_synthetic_health --config configs/default.yaml
"""

__version__ = "1.0.0"
