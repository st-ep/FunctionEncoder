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
                              *args, **kwargs) -> int:
        """Predicts the number of parameters for the Deep Sets encoder."""
        # Calculate input size for phi (concatenated x and y)
        # Assuming x is flattened if multi-dimensional, and y is flattened
        phi_input_dim = np.prod(input_size) + np.prod(output_size)
        phi_output_dim = phi_hidden_size # Output of phi is typically a hidden representation

        # Phi MLP parameters
        n_params_phi = (phi_input_dim + 1) * phi_hidden_size # Input layer
        if phi_n_layers > 2:
            n_params_phi += (phi_n_layers - 2) * (phi_hidden_size + 1) * phi_hidden_size # Hidden layers
        if phi_n_layers > 1:
             n_params_phi += (phi_hidden_size + 1) * phi_output_dim # Output layer
        else: # phi_n_layers == 1
             n_params_phi = (phi_input_dim + 1) * phi_output_dim


        # Rho MLP parameters
        # Input to rho is the output of phi (after aggregation, so same dim)
        rho_input_dim = phi_output_dim
        rho_output_dim = n_basis # Final output is the representation vector

        n_params_rho = (rho_input_dim + 1) * rho_hidden_size # Input layer
        if rho_n_layers > 2:
            n_params_rho += (rho_n_layers - 2) * (rho_hidden_size + 1) * rho_hidden_size # Hidden layers
        if rho_n_layers > 1:
            n_params_rho += (rho_hidden_size + 1) * rho_output_dim # Output layer
        else: # rho_n_layers == 1
            n_params_rho = (rho_input_dim + 1) * rho_output_dim

        return n_params_phi + n_params_rho

    def __init__(self,
                 input_size: tuple[int],
                 output_size: tuple[int],
                 n_basis: int,
                 phi_hidden_size: int = 128,
                 phi_n_layers: int = 3,
                 rho_hidden_size: int = 128,
                 rho_n_layers: int = 3,
                 activation: str = "relu",
                 aggregation: str = "mean", # 'mean', 'sum', 'max'
                 *args, **kwargs): # Consume unused args like learn_basis_functions
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.aggregation = aggregation

        phi_input_dim = np.prod(input_size) + np.prod(output_size)
        phi_output_dim = phi_hidden_size
        rho_input_dim = phi_output_dim
        rho_output_dim = n_basis

        # Build Phi network (processes individual points)
        phi_layers = []
        current_dim = phi_input_dim
        if phi_n_layers == 1:
             phi_layers.append(torch.nn.Linear(current_dim, phi_output_dim))
        else:
            phi_layers.append(torch.nn.Linear(current_dim, phi_hidden_size))
            phi_layers.append(get_activation(activation))
            for _ in range(phi_n_layers - 2):
                phi_layers.append(torch.nn.Linear(phi_hidden_size, phi_hidden_size))
                phi_layers.append(get_activation(activation))
            phi_layers.append(torch.nn.Linear(phi_hidden_size, phi_output_dim))
        self.phi = torch.nn.Sequential(*phi_layers)


        # Build Rho network (processes aggregated representation)
        rho_layers = []
        current_dim = rho_input_dim
        if rho_n_layers == 1:
             rho_layers.append(torch.nn.Linear(current_dim, rho_output_dim))
        else:
            rho_layers.append(torch.nn.Linear(current_dim, rho_hidden_size))
            rho_layers.append(get_activation(activation))
            for _ in range(rho_n_layers - 2):
                rho_layers.append(torch.nn.Linear(rho_hidden_size, rho_hidden_size))
                rho_layers.append(get_activation(activation))
            rho_layers.append(torch.nn.Linear(rho_hidden_size, rho_output_dim))
        self.rho = torch.nn.Sequential(*rho_layers)

        # Verify params
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(
            input_size, output_size, n_basis,
            phi_hidden_size, phi_n_layers, rho_hidden_size, rho_n_layers, activation
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
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation}")
        # Shape: (n_functions, phi_output_dim)

        # Apply rho to the aggregated representation
        representation = self.rho(aggregated)
        # Shape: (n_functions, n_basis)

        return representation 