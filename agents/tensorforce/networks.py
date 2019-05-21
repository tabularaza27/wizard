"""Describes networks in tensorforce specification for use in multiple agents"""

from agents.rl_agent import STATE_DIMENSIONS, ACTION_DIMENSIONS

no_layer = []

small_single_layer = [
    dict(type='dense', size=64)
]

default_single_layer = [
    dict(type='dense', size=ACTION_DIMENSIONS
        + (STATE_DIMENSIONS - ACTION_DIMENSIONS) / 2)
]

default_2_layer = [
    dict(type='dense', size=ACTION_DIMENSIONS
        + 2 * (STATE_DIMENSIONS - ACTION_DIMENSIONS) / 3),
    dict(type='dense', size=ACTION_DIMENSIONS
        + (STATE_DIMENSIONS - ACTION_DIMENSIONS) / 3)
]
