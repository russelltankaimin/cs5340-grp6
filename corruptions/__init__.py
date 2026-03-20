"""
File: corruptions/__init__.py
Description: Initialises the corruptions package. By importing the submodules here, 
we ensure that all @register_corruption decorators are executed and the 
CORRUPTION_REGISTRY is fully populated whenever the package is accessed.
"""

from .registry import CORRUPTION_REGISTRY, register_corruption

# Explicitly import the modules so their decorators fire
from . import additive
from . import frequency
from . import nonlinear