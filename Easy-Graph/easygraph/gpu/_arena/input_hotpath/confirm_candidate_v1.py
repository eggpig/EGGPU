"""Final confirmation composition; frozen components remain unchanged."""
from easygraph.gpu._arena.input_hotpath.owned_csr_v6 import OwnedCandidateAdapter
from easygraph.gpu._arena.sssp_ieee_session_v3 import install_device_delta


def detach_ieee_hooks(session):
    # Instance-bound methods otherwise form a self-reference cycle. Release
    # only this installer's hooks after the normal charged session close.
    for name in ('_query_columns', '_pre_ieee_query_columns'):
        session.__dict__.pop(name, None)


class ConfirmCandidateAdapter(OwnedCandidateAdapter):
    def __init__(self, data, *, budget_bytes, **kwargs):
        super().__init__(data, budget_bytes=budget_bytes, **kwargs)
        install_device_delta(self.session)

    def close(self):
        try:
            return super().close()
        finally:
            detach_ieee_hooks(self.session)
