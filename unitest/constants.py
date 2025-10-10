"""Shared regression-test configuration values.

This module gathers the frequency-band definitions and keyword arguments used
throughout the regression helpers and tutorials.  Keeping the values in a
single place ensures the datasets, tests, and documentation examples all rely
on an identical configuration without duplicating the literals in multiple
files.
"""

from __future__ import annotations

FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30.0],
}

FUNCS_PARAMS = {
    "pow_freq_bands__normalize": False,
    "pow_freq_bands__ratios": "all",
    "pow_freq_bands__psd_method": "fft",
    "pow_freq_bands__freq_bands": FREQ_BANDS,
}

__all__ = [
    "FREQ_BANDS",
    "FUNCS_PARAMS",
]

