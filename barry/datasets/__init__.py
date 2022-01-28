"""
======
Models
======

.. currentmodule:: barry.datasets

Generic
=======

.. autosummary::
   :toctree: generated/

   Dataset -- Generic dataset
   MultiDataset -- Multiple datasets combined

Concrete
========

.. autosummary::
   :toctree: generated/

   PowerSpectrum_SDSS_DR12
   PowerSpectrum_Beutler2019
   DummyCorrelationFunction_SDSS_DR12
   DummyPowerSpectrum_ROSS_DR12
   CorrelationFunction_ROSS_DR12
"""

from barry.datasets.dataset import Dataset, MultiDataset
from barry.datasets.dataset_correlation_function import (
    # CorrelationFunction_SDSS_DR12_Z061_NGC,
    CorrelationFunction_ROSS_DR12,
)
from barry.datasets.dataset_power_spectrum import (
    PowerSpectrum_SDSS_DR12,
    PowerSpectrum_Beutler2019,
)
from barry.datasets.dummy import DummyPowerSpectrum_SDSS_DR12, DummyCorrelationFunction_ROSS_DR12

__all__ = [
    "PowerSpectrum_SDSS_DR12",
    "PowerSpectrum_Beutler2019",
    # "CorrelationFunction_SDSS_DR12_Z061_NGC",
    "CorrelationFunction_ROSS_DR12",
    "DummyPowerSpectrum_SDSS_DR12",
    "DummyCorrelationFunction_ROSS_DR12",
]
