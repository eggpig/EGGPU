"""Independent cold-only and cold-plus-typed-update session adapters."""
from easygraph.gpu._arena.round8_host.session import Session
from easygraph.gpu._arena.round8_host.bulk import TypedSession
from easygraph.gpu._arena.round8_cold.authority import ColdStore


class ColdSession(Session):
    store_type = ColdStore

    def stats(self):
        return dict(super().stats(), cold_preparation=self.journal.stats()['cold_preparation'])


class ColdTypedSession(TypedSession):
    store_type = ColdStore

    def stats(self):
        return dict(super().stats(), cold_preparation=self.journal.stats()['cold_preparation'])
