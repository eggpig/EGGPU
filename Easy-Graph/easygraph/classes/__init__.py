import os

from .directed_graph import DiGraph
from .directed_graph import DiGraphC
from .directed_multigraph import MultiDiGraph
from .eggpu_bulk_graph import EGGPUBulkGraph
from .eggpu_bulk_graph import read_eggpu_csr
from .graph import Graph
from .graph import GraphC
from .graphviews import *
from .multigraph import MultiGraph
from .operation import *


try:
    from .base import BaseHypergraph
    from .base import load_structure
    from .hypergraph import Hypergraph
except:
    if os.environ.get("EASYGRAPH_SHOW_OPTIONAL_IMPORT_WARNINGS", "").strip().upper() in {"1", "TRUE", "ON", "YES"}:
        print(
            "Warning raise in module:classes. Please install Pytorch before you use"
            " functions related to Hypergraph"
        )
