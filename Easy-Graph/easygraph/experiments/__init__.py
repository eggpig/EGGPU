import os

try:
    from .base import BaseTask
    from .hypergraphs import HypergraphVertexClassificationTask


except:
    if os.environ.get("EASYGRAPH_SHOW_OPTIONAL_IMPORT_WARNINGS", "").strip().upper() in {"1", "TRUE", "ON", "YES"}:
        print(
            "Warning raise in module: experiments. Please install Pytorch before you use"
            " functions related to nueral network"
        )
