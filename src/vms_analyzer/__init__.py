"""
VMS Analyzer Library
~~~~~~~~~~~~~~~~~~~

A comprehensive library for analyzing VMS (Vessel Monitoring System) data.
"""

from ._version import version as __version__
from .analyzer import VMSAnalyzer

__all__ = ['VMSAnalyzer', '__version__']