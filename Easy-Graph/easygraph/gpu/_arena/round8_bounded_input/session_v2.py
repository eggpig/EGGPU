"""Explicit bounded-input V2 composition; final execution and scratch reclamation are inherited."""
from functools import partial
import numpy as np
from easygraph.gpu._arena.round8_cold.session import ColdSession,ColdTypedSession
from easygraph.gpu._arena.round8_bounded_input.authority import DEFAULT_LOOKUP_BUDGET
from easygraph.gpu._arena.round8_bounded_input.authority_v2 import BoundedInputStoreV2 as BoundedInputStore
from easygraph.gpu._arena.round8_changes.authority import ChangePreparationMixin,PreparedStore,DEFAULT_CHANGE_BUDGET
from easygraph.gpu._arena.round8_lazy.session import LazySession,FinalSession
from easygraph.gpu._arena.round8_host.bulk import TypedSession

class BoundedInputSession(ColdSession):
    store_type=BoundedInputStore

class BoundedInputTypedSession(ColdTypedSession):
    store_type=BoundedInputStore


class BoundedPreparedStore(ChangePreparationMixin,BoundedInputStore):
    pass


class BoundedSession(FinalSession):
    def __init__(self,*args,fast_input=True,lazy_module=True,
                 change_budget_bytes=DEFAULT_CHANGE_BUDGET,
                 input_lut_budget_bytes=DEFAULT_LOOKUP_BUDGET,
                 packed_input_sort=True,**kwargs):
        if not isinstance(fast_input,bool):raise ValueError('fast_input must be bool')
        if isinstance(input_lut_budget_bytes,bool) or not isinstance(input_lut_budget_bytes,(int,np.integer)) or input_lut_budget_bytes<0:
            raise ValueError('Nonnegative integer temporary lookup-table budget required')
        if not isinstance(packed_input_sort,bool):raise ValueError('Boolean packed sort option required')
        options=dict(change_budget_bytes=change_budget_bytes)
        if fast_input:options.update(input_lut_budget_bytes=int(input_lut_budget_bytes),packed_input_sort=packed_input_sort)
        store=BoundedPreparedStore if fast_input else PreparedStore
        LazySession.__init__(self,*args,lazy_module=lazy_module,store_factory=partial(store,**options),**kwargs)
        self.fast_input=fast_input
        self.input_lut_budget_bytes=int(input_lut_budget_bytes);self.packed_input_sort=packed_input_sort

    def stats(self):
        result=super().stats();bounded=self.journal.stats().get('bounded_input')
        result['bounded_input']=(dict(bounded,active=True) if bounded is not None else
            dict(active=False,lookup_table_budget_bytes=self.input_lut_budget_bytes,
                 packed_sort=self.packed_input_sort,lookup_table_live_bytes=0,lookup_table_peak_bytes=0,
                 explicit_temporary_peak_bytes=0,scope='fast_input=False uses the frozen generic cold preparation path'))
        return result


class BoundedTypedSession(BoundedSession,TypedSession):
    pass
