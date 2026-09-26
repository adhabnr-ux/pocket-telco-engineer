"""
Link configuration for the OFDM channel estimation task.

Everything an estimator is allowed to know about the link lives here.
The estimator does NOT know which propagation scenario (delay spread,
Doppler, LOS/NLOS) produced a given batch: it has to be robust across all
of them ("statistics-agnostic" channel estimation).

This file is copied into the workspace next to the candidate ``draft.py``,
so candidates can simply ``from link_config import ...``.
"""

import numpy as np

# --- Resource grid --------------------------------------------------------
NUM_OFDM_SYMBOLS = 14          # one 5G NR slot
FFT_SIZE = 72                  # 6 resource blocks x 12 subcarriers
SUBCARRIER_SPACING = 30e3      # [Hz] numerology mu = 1
CYCLIC_PREFIX_LENGTH = 6       # [samples] (cyclic prefix is removed by the channel model)
CARRIER_FREQUENCY = 3.5e9      # [Hz] n78 mid-band

# OFDM symbol duration including cyclic prefix [s]
OFDM_SYMBOL_DURATION = (1.0 + CYCLIC_PREFIX_LENGTH / FFT_SIZE) / SUBCARRIER_SPACING

# --- Pilots ---------------------------------------------------------------
# Two DMRS-like pilot OFDM symbols, comb-2 in frequency (every other
# subcarrier), similar to 5G NR DMRS configuration type 1 with one
# additional DMRS position.
PILOT_OFDM_SYMBOLS = (2, 11)
PILOT_SUBCARRIER_OFFSET = 0     # pilots on subcarriers 0, 2, 4, ...
PILOT_SUBCARRIER_SPACING = 2

_PILOT_SEED = 1234


def _build_pilots():
    mask = np.zeros((NUM_OFDM_SYMBOLS, FFT_SIZE), dtype=bool)
    for s in PILOT_OFDM_SYMBOLS:
        mask[s, PILOT_SUBCARRIER_OFFSET::PILOT_SUBCARRIER_SPACING] = True
    rng = np.random.default_rng(_PILOT_SEED)
    n = int(mask.sum())
    # Unit-energy QPSK pilots
    qpsk = (rng.choice([-1.0, 1.0], n) + 1j * rng.choice([-1.0, 1.0], n)) / np.sqrt(2.0)
    grid = np.zeros((NUM_OFDM_SYMBOLS, FFT_SIZE), dtype=np.complex64)
    grid[mask] = qpsk.astype(np.complex64)   # row-major (symbol, subcarrier) order
    return mask, grid


# PILOT_MASK[s, k]  is True where resource element (s, k) carries a pilot.
# PILOT_SYMBOLS[s, k] is the known transmitted pilot value (0 elsewhere).
PILOT_MASK, PILOT_SYMBOLS = _build_pilots()
NUM_PILOTS = int(PILOT_MASK.sum())

# Data resource elements carry unit-energy QAM symbols unknown to the estimator.
DATA_NUM_BITS_PER_SYMBOL = 4   # 16-QAM
