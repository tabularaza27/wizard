from agents.rl_agent import STATE_DIMENSIONS, ACTION_DIMENSIONS

input_dimensions = STATE_DIMENSIONS
output_dimensions = ACTION_DIMENSIONS
diff = input_dimensions - output_dimensions

def equal_spacing_fc(num_hidden_layers):
    return tuple(int(input_dimensions - layer * diff / (num_hidden_layers + 1))
        for layer in range(1, num_hidden_layers + 1))
