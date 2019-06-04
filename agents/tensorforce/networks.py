"""Describes networks in tensorforce specification for use in multiple agents"""

from agents.rl_agent import STATE_DIMENSIONS, ACTION_DIMENSIONS

def equal_spacing(num_hidden_layers):
    input_dimensions = STATE_DIMENSIONS
    output_dimensions = ACTION_DIMENSIONS
    diff = input_dimensions - output_dimensions

    layers = []
    for layer in range(1, num_hidden_layers + 1):
        layers.append(dict(type='dense',
            size=int(input_dimensions
                - layer * diff / (num_hidden_layers + 1))))
    return layers

no_layer = []
default_single_layer = equal_spacing(1)
default_2_layer = equal_spacing(2)

small_single_layer = [
    dict(type='dense', size=64)
]
