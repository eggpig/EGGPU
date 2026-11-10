"""Explicit readonly native FP64 SSSP/weighted-PR snapshot-view lifetime.

This holds actual resident input views, not public result arrays. It permits
completed-query interleaving and never claims a concurrently executing kernel.
Warm both weighted functions first; close every lease before closing Session.
"""
from threading import RLock
import weakref


class ReadOnlySnapshotLease:
    def __init__(self,session,snapshot,handle):
        self._session=weakref.ref(session);self._session_id=session.session_id
        self.snapshot=snapshot;self._handle=handle;self._lock=RLock();self.closed=False

    def _owner(self):
        session=self._session()
        if session is None or session.session_id!=self._session_id:
            raise RuntimeError('Snapshot lease Session no longer exists')
        return session

    def read_weights(self):
        """Return owning readonly host arrays for full bitwise qualification."""
        with self._lock:
            if self.closed:raise RuntimeError('Snapshot lease is closed')
            session=self._owner()
            with session._lock:
                session._open()
                return self._handle.read_weights(self._session_id)

    def close(self):
        with self._lock:
            if self.closed:return
            session=self._owner()
            with session._lock:
                # Native close explicitly reports an ended/replaced runtime;
                # silently succeeding here would conceal wrong shutdown order.
                self._handle.close(self._session_id)
                self.closed=True

    def __enter__(self):
        if self.closed:raise RuntimeError('Snapshot lease is closed')
        return self

    def __exit__(self,*_):self.close()


def acquire_snapshot_lease(session,snapshot):
    """Hold the named existing FP64 input views under Session's request lock."""
    with session._lock:
        session._open()
        if session.kind!='native':raise ValueError('Readonly snapshot lease requires native Session')
        backend=session._backend
        if snapshot not in backend.graphs:raise ValueError('Unknown snapshot')
        method=getattr(backend.cpp,'cpp_egmgpu_acquire_readonly_snapshot_lease',None)
        if method is None:raise RuntimeError('Native DSO lacks readonly snapshot lease ABI')
        native=method(backend.graphs[snapshot],session.session_id,devices=backend.devices)
        try:return ReadOnlySnapshotLease(session,snapshot,native)
        except BaseException:
            native.close(session.session_id)
            raise
