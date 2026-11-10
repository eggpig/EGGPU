"""Explicit GPU sessions; importing this package does not start CUDA.

``Session.native`` uses one implementation for one or many devices.
``Session.arena`` retains the bounded single-device mutable-graph executor.
``Session.create`` selects between these using an inspectable capability policy.
Both execute version-bound operator requests and return owning host outputs.
"""

from .session import Session, OperatorRequest, OperatorSpec, StaleRequestError
from .contracts import QueryContract, GraphDescriptor, GraphVersion
from ._dispatch import SessionPlan, plan_session

__all__ = ["Session", "OperatorRequest", "OperatorSpec", "StaleRequestError",
           "QueryContract", "GraphDescriptor", "GraphVersion", "SessionPlan", "plan_session"]
