import os

from .deepwalk import *
from .NOBE import *
from .node2vec import *


try:
    from .line import *
    from .sdne import *
except:
    if os.environ.get("EASYGRAPH_SHOW_OPTIONAL_IMPORT_WARNINGS", "").strip().upper() in {"1", "TRUE", "ON", "YES"}:
        print(
            "Warning raise in module:graph_embedding. Please install packages Pytorch"
            " before you use functions related to graph_embedding"
        )
