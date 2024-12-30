import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture


# Returns the desired activation function by name
def get_activation( activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class MLP(BaseArchitecture):


    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=256, n_layers=4, learn_basis_functions=True, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        n_params =  input_size * hidden_size + hidden_size + \
                    (n_layers - 2) * hidden_size * hidden_size + (n_layers - 2) * hidden_size + \
                    hidden_size * output_size + output_size
        return n_params

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=256,
                 n_layers:int=4,
                 activation:str="relu",
                 learn_basis_functions=True):
        super(MLP, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions


        # get inputs
        input_size = input_size[0]  # only 1D input supported for now
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]

        # build net
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, learn_basis_functions=learn_basis_functions)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # this is the main part of this function. The rest is just error handling
        outs = self.model(x)
        if self.learn_basis_functions:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs
        
    def compute_orthogonality_penalty(
        self, 
        x: torch.Tensor, 
        weight: float = 1.0,
        enforce_unit_norm: bool = True,
    ) -> torch.Tensor:
        """
        Computes a penalty encouraging the learned basis functions to be
        orthogonal (and optionally unit norm).

        Args:
            x (torch.Tensor): input data, shape (f, d, input_dim) or (d, input_dim).
            weight (float): scaling factor for the penalty.
            enforce_unit_norm (bool): True => push diagonal of Gram to 1,
                                      False => only push off-diagonals to 0.

        Returns:
            torch.Tensor: a scalar penalty (requires_grad=True).
        """
        # Forward pass => shape (f, d, output_dim, n_basis) if learn_basis_functions is True
        Gs = self.forward(x)

        # Flatten all but the last dimension (n_basis). E.g. if shape is (f, d, 1, k),
        # we want shape (f*d, k) or (f*d*m, k) in a more general scenario.
        if Gs.dim() == 4:
            # e.g. (f, d, 1, k)
            Gs = Gs.reshape(-1, Gs.shape[-1])  # (f*d*m, k)
        elif Gs.dim() == 3:
            # e.g. (f, d, k) if output_size=(1,)
            Gs = Gs.reshape(-1, Gs.shape[-1])  # (f*d, k)
        else:
            # If you've got a different shape, adapt accordingly
            raise ValueError(f"Unexpected shape for Gs: {Gs.shape}")

        # Now Gs is (N, k), where N = f*d*(m...) and k = n_basis
        # Gram matrix: shape (k, k)
        Gram = Gs.T @ Gs  # i.e., G^T G

        # We want Gram ~ I for orthonormal. 
        # If `enforce_unit_norm=True`, penalize full (Gram - I). 
        # Otherwise, penalize only off-diagonals (orthogonality but not unit norm).
        I = torch.eye(Gram.shape[0], device=Gram.device)
        if enforce_unit_norm:
            # Full orthonormal penalty (off-diagonal + diagonal => I)
            penalty = (Gram - I).pow(2).sum()
        else:
            # Only orthogonality => zero out diagonal, penalize off-diagonals
            diag = torch.diag(torch.diag(Gram))
            penalty = (Gram - diag).pow(2).sum()

        return weight * penalty
        


