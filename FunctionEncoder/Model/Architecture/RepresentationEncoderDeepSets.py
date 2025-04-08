import torch
from typing import Tuple
import numpy as np

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from .utils import get_activation, ParallelLinear

class RepresentationEncoderDeepSets(BaseArchitecture):
    """
    A Representation Encoder using the Deep Sets architecture.
    It processes each (x_i, y_i) pair with a network (phi), aggregates the results,
    and then processes the aggregated representation with another network (rho).
    """

    @staticmethod
    def predict_number_params(input_size: tuple[int],
                              output_size: tuple[int],
                              n_basis: int,
                              phi_hidden_size: int = 128,
                              phi_n_layers: int = 3,
                              rho_hidden_size: int = 128,
                              rho_n_layers: int = 3,
                              activation: str = "relu",
                              aggregation: str = "mean", # 'mean', 'sum', 'max', 'attention'
                              use_layer_norm: bool = False, # New flag
                              *args, **kwargs) -> int:
        """Predicts the number of parameters for the Deep Sets encoder."""
        # Calculate input size for phi (concatenated x and y)
        # Assuming x is flattened if multi-dimensional, and y is flattened
        phi_input_dim = np.prod(input_size) + np.prod(output_size)
        phi_output_dim = phi_hidden_size # Output of phi is typically a hidden representation

        # Phi MLP parameters
        n_params_phi = 0
        n_params_phi_ln = 0
        current_dim = phi_input_dim
        if phi_n_layers == 1:
            n_params_phi += (current_dim + 1) * phi_output_dim
        else:
            # Input layer
            n_params_phi += (current_dim + 1) * phi_hidden_size
            if use_layer_norm:
                n_params_phi_ln += 2 * phi_hidden_size # LayerNorm after first activation
            # Hidden layers
            for _ in range(phi_n_layers - 2):
                n_params_phi += (phi_hidden_size + 1) * phi_hidden_size
                if use_layer_norm:
                    n_params_phi_ln += 2 * phi_hidden_size # LayerNorm after hidden activations
            # Output layer
            n_params_phi += (phi_hidden_size + 1) * phi_output_dim
            # No LayerNorm after the final linear layer of phi

        # Rho MLP parameters
        # Input to rho is the output of phi (after aggregation, so same dim)
        rho_input_dim = phi_output_dim
        rho_output_dim = n_basis # Final output is the representation vector

        n_params_rho = 0
        n_params_rho_ln = 0
        current_dim = rho_input_dim
        if rho_n_layers == 1:
            n_params_rho += (current_dim + 1) * rho_output_dim
        else:
            # Input layer
            n_params_rho += (current_dim + 1) * rho_hidden_size
            if use_layer_norm:
                n_params_rho_ln += 2 * rho_hidden_size # LayerNorm after first activation
            # Hidden layers
            for _ in range(rho_n_layers - 2):
                n_params_rho += (rho_hidden_size + 1) * rho_hidden_size
                if use_layer_norm:
                    n_params_rho_ln += 2 * rho_hidden_size # LayerNorm after hidden activations
            # Output layer
            n_params_rho += (rho_hidden_size + 1) * rho_output_dim
            # No LayerNorm after the final linear layer of rho

        # Attention network parameters (if applicable)
        n_params_attention = 0
        if aggregation == "attention":
            # Simple linear layer: phi_output_dim -> 1
            n_params_attention = (phi_output_dim + 1) * 1

        return n_params_phi + n_params_rho + n_params_attention + n_params_phi_ln + n_params_rho_ln

    def __init__(self,
                 input_size: tuple[int],
                 output_size: tuple[int],
                 n_basis: int,
                 phi_hidden_size: int = 128,
                 phi_n_layers: int = 3,
                 rho_hidden_size: int = 128,
                 rho_n_layers: int = 3,
                 activation: str = "relu",
                 aggregation: str = "mean", # 'mean', 'sum', 'max', 'attention'
                 use_layer_norm: bool = False, # New flag
                 *args, **kwargs): # Consume unused args like learn_basis_functions
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        if aggregation not in ["mean", "sum", "max", "attention"]:
             raise ValueError(f"Unknown aggregation type: {aggregation}")
        self.aggregation = aggregation
        self.use_layer_norm = use_layer_norm # Store the flag

        phi_input_dim = np.prod(input_size) + np.prod(output_size)
        phi_output_dim = phi_hidden_size
        rho_input_dim = phi_output_dim
        rho_output_dim = n_basis

        # Build Phi network (processes individual points)
        phi_layers = []
        current_dim = phi_input_dim
        if phi_n_layers == 1:
             phi_layers.append(torch.nn.Linear(current_dim, phi_output_dim))
             # No activation or LN needed for single layer
        else:
            # Input layer
            phi_layers.append(torch.nn.Linear(current_dim, phi_hidden_size))
            phi_layers.append(get_activation(activation))
            if self.use_layer_norm:
                phi_layers.append(torch.nn.LayerNorm(phi_hidden_size))
            # Hidden layers
            for _ in range(phi_n_layers - 2):
                phi_layers.append(torch.nn.Linear(phi_hidden_size, phi_hidden_size))
                phi_layers.append(get_activation(activation))
                if self.use_layer_norm:
                    phi_layers.append(torch.nn.LayerNorm(phi_hidden_size))
            # Output layer
            phi_layers.append(torch.nn.Linear(phi_hidden_size, phi_output_dim))
            # No activation or LN after the final linear layer
        self.phi = torch.nn.Sequential(*phi_layers)

        # Build Attention network (if using attention aggregation)
        self.attention_net = None
        if self.aggregation == "attention":
            self.attention_net = torch.nn.Linear(phi_output_dim, 1)

        # Build Rho network (processes aggregated representation)
        rho_layers = []
        current_dim = rho_input_dim
        if rho_n_layers == 1:
             rho_layers.append(torch.nn.Linear(current_dim, rho_output_dim))
             # No activation or LN needed for single layer
        else:
            # Input layer
            rho_layers.append(torch.nn.Linear(current_dim, rho_hidden_size))
            rho_layers.append(get_activation(activation))
            if self.use_layer_norm:
                rho_layers.append(torch.nn.LayerNorm(rho_hidden_size))
            # Hidden layers
            for _ in range(rho_n_layers - 2):
                rho_layers.append(torch.nn.Linear(rho_hidden_size, rho_hidden_size))
                rho_layers.append(get_activation(activation))
                if self.use_layer_norm:
                    rho_layers.append(torch.nn.LayerNorm(rho_hidden_size))
            # Output layer
            rho_layers.append(torch.nn.Linear(rho_hidden_size, rho_output_dim))
            # No activation or LN after the final linear layer
        self.rho = torch.nn.Sequential(*rho_layers)

        # Verify params
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(
            input_size, output_size, n_basis,
            phi_hidden_size, phi_n_layers, rho_hidden_size, rho_n_layers, activation, aggregation, use_layer_norm # Pass new flag
        )
        assert n_params == estimated_n_params, f"Encoder has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, example_xs: torch.Tensor, example_ys: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Deep Sets encoder.

        Args:
            example_xs: Example inputs. Shape (n_functions, n_datapoints, *input_size)
            example_ys: Example outputs. Shape (n_functions, n_datapoints, *output_size)

        Returns:
            Representation tensor. Shape (n_functions, n_basis)
        """
        n_functions = example_xs.shape[0]
        n_datapoints = example_xs.shape[1]

        # Flatten input_size and output_size dimensions
        xs_flat = example_xs.view(n_functions, n_datapoints, -1)
        ys_flat = example_ys.view(n_functions, n_datapoints, -1)

        # Concatenate x and y for each point
        # Shape: (n_functions, n_datapoints, flattened_input_size + flattened_output_size)
        combined = torch.cat([xs_flat, ys_flat], dim=-1)

        # Apply phi to each point
        # Reshape for phi: (n_functions * n_datapoints, combined_dim)
        phi_input = combined.view(-1, combined.shape[-1])
        phi_output = self.phi(phi_input)
        # Reshape back: (n_functions, n_datapoints, phi_output_dim)
        phi_output = phi_output.view(n_functions, n_datapoints, -1)

        # Aggregate phi outputs
        if self.aggregation == "mean":
            aggregated = torch.mean(phi_output, dim=1)
        elif self.aggregation == "sum":
            aggregated = torch.sum(phi_output, dim=1)
        elif self.aggregation == "max":
            aggregated = torch.max(phi_output, dim=1)[0]
        elif self.aggregation == "attention":
            assert self.attention_net is not None, "Attention network not initialized for attention aggregation."
            # Compute attention scores
            # Input shape: (n_functions * n_datapoints, phi_output_dim)
            attention_scores = self.attention_net(phi_output.view(-1, phi_output.shape[-1]))
            # Reshape scores: (n_functions, n_datapoints, 1)
            attention_scores = attention_scores.view(n_functions, n_datapoints, 1)
            # Normalize scores using softmax across datapoints
            attention_weights = torch.softmax(attention_scores, dim=1)
            # Apply attention weights (weighted sum)
            # phi_output: (n_functions, n_datapoints, phi_output_dim)
            # attention_weights: (n_functions, n_datapoints, 1)
            # Result: (n_functions, phi_output_dim)
            aggregated = torch.sum(phi_output * attention_weights, dim=1)
        else:
            # This case should not be reached due to check in __init__
            raise ValueError(f"Unknown aggregation type: {self.aggregation}")
        # Shape: (n_functions, phi_output_dim)

        # Apply rho to the aggregated representation
        representation = self.rho(aggregated)
        # Shape: (n_functions, n_basis)

        return representation 