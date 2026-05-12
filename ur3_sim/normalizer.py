"""Small normalizer copy compatible with the upstream UR3 wrapper."""

import json
from pathlib import Path

import numpy as np


class IdentityNormalizer:
    """Pass-through normalizer used by the VersatIL server boundary."""

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return state

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action


class MinMaxNormalizer:
    """Upstream min-max normalizer for optional normalized action/state I/O."""

    def __init__(self, normalize_actions_flag: bool = True):
        self.state_stats: dict = {}
        self.action_stats: dict = {}
        self.goal_stats: dict = {}
        self.normalize_actions_flag = normalize_actions_flag

    def load_stats(self, path: str | Path) -> None:
        with open(path, "r") as file:
            stats = json.load(file)
        self.state_stats = stats["state_stats"]
        self.action_stats = stats["action_stats"]
        self.goal_stats = stats.get("goal_stats", {})

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if not self.state_stats:
            return state
        min_value = np.asarray(self.state_stats["min"], dtype=np.float64)
        max_value = np.asarray(self.state_stats["max"], dtype=np.float64)
        return 2 * (state - min_value) / (max_value - min_value + 1e-8) - 1

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions_flag or not self.action_stats:
            return normalized_action
        min_value = np.asarray(self.action_stats["min"], dtype=np.float64)
        max_value = np.asarray(self.action_stats["max"], dtype=np.float64)
        return (normalized_action + 1) * (max_value - min_value + 1e-8) / 2 + min_value
