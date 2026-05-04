"""Project-specific constants for the UR3 block-pushing server."""

from enum import Enum


DEFAULT_CLIENT_NAME = "unknown"
MAX_STEPS = 1000
NO_OP_ACTION = [0.45, -0.325]


class UR3TrajectoryColumn(str, Enum):
    """Column names for trajectory CSV recording."""

    EE_POS_X = "ee_pos_x"
    EE_POS_Y = "ee_pos_y"
    BLOCK1_POS_X = "block1_pos_x"
    BLOCK1_POS_Y = "block1_pos_y"
    BLOCK2_POS_X = "block2_pos_x"
    BLOCK2_POS_Y = "block2_pos_y"
    ACTION_X = "action_x"
    ACTION_Y = "action_y"
