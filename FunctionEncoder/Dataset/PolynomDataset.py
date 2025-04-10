from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class PolynomDataset(BaseDataset):
    """
    Dataset for generating samples from fifth-order polynomials:
    y = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f
    """

    def __init__(self,
                 a_range=(-1, 1),
                 b_range=(-1, 1),
                 c_range=(-1, 1),
                 d_range=(-1, 1),
                 e_range=(-1, 1),
                 f_range=(-1, 1),
                 input_range=(-5, 5),
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_functions:int=None,
                 n_examples:int=None,
                 n_queries:int=None,

                 # deprecated arguments
                 n_functions_per_sample:int = None,
                 n_examples_per_sample:int = None,
                 n_points_per_sample:int = None,
                 ):
        # Handle default arguments and deprecated ones
        if n_functions is None and n_functions_per_sample is None:
            n_functions = 10
        if n_examples is None and n_examples_per_sample is None:
            n_examples = 1000
        if n_queries is None and n_points_per_sample is None:
            n_queries = 10000

        super().__init__(input_size=(1,),
                         output_size=(1,),
                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,

                         # deprecated arguments
                         total_n_functions=None,
                         total_n_samples_per_function=None,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )

        # Store coefficient ranges as tensors
        self.a_range = torch.tensor(a_range, device=self.device, dtype=self.dtype)
        self.b_range = torch.tensor(b_range, device=self.device, dtype=self.dtype)
        self.c_range = torch.tensor(c_range, device=self.device, dtype=self.dtype)
        self.d_range = torch.tensor(d_range, device=self.device, dtype=self.dtype)
        self.e_range = torch.tensor(e_range, device=self.device, dtype=self.dtype)
        self.f_range = torch.tensor(f_range, device=self.device, dtype=self.dtype)
        self.input_range = torch.tensor(input_range, device=self.device, dtype=self.dtype)

    def sample(self) -> Tuple[  torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                dict]:
        """
        Samples a batch of functions, examples, and queries.
        """
        with torch.no_grad():
            n_functions = self.n_functions
            n_examples = self.n_examples
            n_queries = self.n_queries

            # Generate n_functions sets of coefficients (A, B, C, D, E, F)
            As = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
            Bs = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
            Cs = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
            Ds = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.d_range[1] - self.d_range[0]) + self.d_range[0]
            Es = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.e_range[1] - self.e_range[0]) + self.e_range[0]
            Fs = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.f_range[1] - self.f_range[0]) + self.f_range[0]

            # Generate example and query points (xs)
            query_xs = torch.rand((n_functions, n_queries, *self.input_size), dtype=self.dtype, device=self.device)
            query_xs = query_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            example_xs = torch.rand((n_functions, n_examples, *self.input_size), dtype=self.dtype, device=self.device)
            example_xs = example_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # Compute the corresponding ys using the fifth-order polynomial equation
            # Unsqueeze coefficients to allow broadcasting: (n_functions, 1, 1)
            A_unsqueezed = As.unsqueeze(1)
            B_unsqueezed = Bs.unsqueeze(1)
            C_unsqueezed = Cs.unsqueeze(1)
            D_unsqueezed = Ds.unsqueeze(1)
            E_unsqueezed = Es.unsqueeze(1)
            F_unsqueezed = Fs.unsqueeze(1)

            query_ys = (A_unsqueezed * query_xs**5 +
                        B_unsqueezed * query_xs**4 +
                        C_unsqueezed * query_xs**3 +
                        D_unsqueezed * query_xs**2 +
                        E_unsqueezed * query_xs +
                        F_unsqueezed)

            example_ys = (A_unsqueezed * example_xs**5 +
                          B_unsqueezed * example_xs**4 +
                          C_unsqueezed * example_xs**3 +
                          D_unsqueezed * example_xs**2 +
                          E_unsqueezed * example_xs +
                          F_unsqueezed)

            coefficients = {
                "As": As, "Bs": Bs, "Cs": Cs,
                "Ds": Ds, "Es": Es, "Fs": Fs
            }

            return example_xs, example_ys, query_xs, query_ys, coefficients
