"""
MIST - Microscopy Image Stitching Tool

A high-performance Python library for stitching 2D microscopy image datasets.
Developed at the National Institute of Standards and Technology.

For more information, visit: https://github.com/usnistgov/MIST
"""

__version__ = "1.0.0"
__author__ = "National Institute of Standards and Technology"
__email__ = "mist@nist.gov"
__license__ = "Public Domain"

from .main import mist

__all__ = ["mist"]
