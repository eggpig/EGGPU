"""Qualified structural SSSP dispatch with complete capacity forwarding."""
from easygraph.gpu._arena.sssp_policy import PolicyFrontierSSSP

class QualifiedSSSP(PolicyFrontierSSSP):
    # Session workspace growth updates both storage and accounting capacity.
    # The underlying original native bridge must see the new accounting value.
    @property
    def workspace_bytes(self):return self.base.workspace_bytes
    @workspace_bytes.setter
    def workspace_bytes(self,value):self.base.workspace_bytes=value
