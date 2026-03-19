"""
Allows installing as Python package.
"""
from setuptools import setup

setup(
    name="recommend_gnn",  # The name of your package
    version="0.1.0",         # Version number
    description="Recommend Amazon products with a GNN",  # A short description
    packages=["recommend_gnn"],  # List of package directories
)