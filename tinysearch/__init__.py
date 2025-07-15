"""
TinySearch: A lightweight vector retrieval system
"""

__version__ = "0.1.0"

# Make submodules available for import
from . import adapters
from . import splitters
from . import embedders 
from . import indexers
from . import query
from . import flow 