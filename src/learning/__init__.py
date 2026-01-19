"""Continuous learning modules."""

from .online import OnlineLearner
from .periodic import PeriodicRetrainer
from .ewc import EWCLearner
from .ucl import UCLLearner
from .replay_buffer import ReplayBuffer

__all__ = ["OnlineLearner", "PeriodicRetrainer", "EWCLearner", "UCLLearner", "ReplayBuffer"]
