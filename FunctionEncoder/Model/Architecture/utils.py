import torch
import torch.nn as nn

# Assume get_activation looks something like this:
def get_activation(activation: str) -> nn.Module:
    """Returns the activation function module."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    # Add GELU activation
    elif activation == "gelu":
        return nn.GELU()
    # Add other activations as needed
    else:
        raise ValueError(f"Unknown activation function: {activation}")

# Assume ParallelLinear looks something like this (based on ParallelMLP usage):
class ParallelLinear(nn.Module):
    """Applies N linear layers in parallel."""
    def __init__(self, in_features: int, out_features: int, n_parallel: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        # Create weights and biases for N parallel layers
        # Weight shape: (n_parallel, out_features, in_features)
        # Bias shape: (n_parallel, out_features)
        self.weight = nn.Parameter(torch.Tensor(n_parallel, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(n_parallel, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and biases (e.g., Kaiming uniform)
        for i in range(self.n_parallel):
            nn.init.kaiming_uniform_(self.weight[i], a=torch.math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / torch.math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ParallelLinear.
        Input x shape: (batch_size, n_datapoints, in_features)
                       or (batch_size, n_datapoints, n_parallel, in_features) if already parallel
        Output shape: (batch_size, n_datapoints, n_parallel, out_features)
        """
        # Input might be (f, d, m) or (f, d, m, k) from FunctionEncoder perspective
        # Or more generally (..., in_features) or (..., n_parallel, in_features)

        # We expect input like (..., n_parallel, in_features) for einsum
        # If input is (..., in_features), expand it
        if x.shape[-1] == self.in_features and x.shape[-2] != self.n_parallel:
             # Add n_parallel dimension before the last one
             x = x.unsqueeze(-2).expand(*x.shape[:-1], self.n_parallel, self.in_features)


        # Perform parallel linear transformations using einsum
        # '...ki,koi->...ko'
        # ... represents batch dimensions (e.g., n_functions, n_datapoints)
        # k is n_parallel, i is in_features, o is out_features
        out = torch.einsum('...ki,koi->...ko', x, self.weight) + self.bias.unsqueeze(0).unsqueeze(0) # Add batch dims for bias

        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, n_parallel={self.n_parallel}, bias={self.bias is not None}' 