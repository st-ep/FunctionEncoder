import torch
import numpy as np
from typing import Tuple, List, Dict, Any

from FunctionEncoder.Dataset.BaseDataset import BaseDataset

class NonLinearDataset(BaseDataset):
    """
    Dataset generating non-linear functions based on sinusoids:
    f(x) = amplitude * sin(frequency * x + phase) + vertical_shift
    """
    def __init__(self,
                 n_functions_per_batch: int,
                 n_samples_per_function: int,
                 n_query_per_function: int,
                 input_range: Tuple[float, float] = (-3.0, 3.0),
                 amplitude_range: Tuple[float, float] = (0.5, 2.0),
                 frequency_range: Tuple[float, float] = (0.5, 2.0),
                 phase_range: Tuple[float, float] = (0, 2 * np.pi),
                 v_shift_range: Tuple[float, float] = (-1.0, 1.0),
                 noise_std: float = 0.01, # Standard deviation of Gaussian noise added to y
                 seed: int = None):
        """
        Args:
            n_functions_per_batch: Number of different functions (parameter sets) in a batch.
            n_samples_per_function: Number of (x, y) pairs for the example set (context).
            n_query_per_function: Number of (x, y) pairs for the query set (evaluation).
            input_range: Range from which to sample x values.
            amplitude_range: Range for the sinusoid amplitude.
            frequency_range: Range for the sinusoid frequency.
            phase_range: Range for the sinusoid phase.
            v_shift_range: Range for the vertical shift.
            noise_std: Standard deviation of Gaussian noise added to y values. Set to 0 for no noise.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(n_functions_per_batch, n_samples_per_function, n_query_per_function, seed)

        self.input_range = input_range
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range
        self.phase_range = phase_range
        self.v_shift_range = v_shift_range
        self.noise_std = noise_std

        # Input/Output sizes are fixed for this dataset
        self.input_size = (1,)
        self.output_size = (1,)
        self.data_type = "deterministic" # Even with noise, the underlying function is deterministic

    def sample_function(self) -> Dict[str, float]:
        """Samples parameters for a single sinusoid function."""
        amplitude = self.rng.uniform(*self.amplitude_range)
        frequency = self.rng.uniform(*self.frequency_range)
        phase = self.rng.uniform(*self.phase_range)
        v_shift = self.rng.uniform(*self.v_shift_range)
        return {"amplitude": amplitude, "frequency": frequency, "phase": phase, "v_shift": v_shift}

    def evaluate_function(self, x: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
        """Evaluates the sinusoid function defined by params at input x."""
        y = params["amplitude"] * torch.sin(params["frequency"] * x + params["phase"]) + params["v_shift"]
        if self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y += noise
        return y

    def sample_datapoints(self, n_points: int, params: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples n_points (x, y) pairs for a given function."""
        # Sample x values uniformly
        xs = torch.rand(n_points, *self.input_size, generator=self.torch_rng) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        xs = xs.sort(dim=0).values # Sorting helps visualization but isn't strictly necessary

        # Evaluate function
        ys = self.evaluate_function(xs, params)
        return xs, ys

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Samples a batch of data containing example sets and query sets for multiple functions.

        Returns:
            Tuple: (example_xs, example_ys, query_xs, query_ys, function_params_list)
                - example_xs: (n_functions, n_samples, *input_size)
                - example_ys: (n_functions, n_samples, *output_size)
                - query_xs: (n_functions, n_query, *input_size)
                - query_ys: (n_functions, n_query, *output_size)
                - function_params_list: List of dictionaries, one for each function.
        """
        batch_example_xs = []
        batch_example_ys = []
        batch_query_xs = []
        batch_query_ys = []
        function_params_list = []

        for _ in range(self.n_functions_per_batch):
            # 1. Sample function parameters
            params = self.sample_function()
            function_params_list.append(params)

            # 2. Sample example points for this function
            example_xs, example_ys = self.sample_datapoints(self.n_samples_per_function, params)
            batch_example_xs.append(example_xs)
            batch_example_ys.append(example_ys)

            # 3. Sample query points for this function
            # Ensure query points are distinct from example points if needed,
            # but for continuous domains, resampling is usually sufficient.
            query_xs, query_ys = self.sample_datapoints(self.n_query_per_function, params)
            batch_query_xs.append(query_xs)
            batch_query_ys.append(query_ys)

        # Stack lists into tensors
        example_xs_tensor = torch.stack(batch_example_xs)
        example_ys_tensor = torch.stack(batch_example_ys)
        query_xs_tensor = torch.stack(batch_query_xs)
        query_ys_tensor = torch.stack(batch_query_ys)

        return example_xs_tensor, example_ys_tensor, query_xs_tensor, query_ys_tensor, function_params_list

    def check_dataset(self):
        """Basic check of dataset configuration."""
        assert self.input_size == (1,), "Input size must be (1,) for this dataset"
        assert self.output_size == (1,), "Output size must be (1,) for this dataset"
        assert self.data_type == "deterministic", "Data type should be deterministic"
        print("NonLinearDataset checks passed.") 