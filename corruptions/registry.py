"""
File: corruptions/registry.py
Description: A central registry to manage all differentiable audio corruption functions.
This allows scripts to dynamically load and apply corruptions by their string names.
"""

from typing import Callable, Dict
import torch

# Global dictionary storing registered corruption functions
CORRUPTION_REGISTRY: Dict[str, Callable] = {}

def register_corruption(name: str):
    """
    Decorator function to register a new corruption model into the global registry.
    
    Args:
        name (str): The unique string identifier for the corruption function.
    """
    def decorator(fn: Callable):
        CORRUPTION_REGISTRY[name] = fn
        return fn
    return decorator