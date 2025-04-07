from typing import Union, Tuple
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.Euclidean import Euclidean
from FunctionEncoder.Model.Architecture.MLP import MLP
from FunctionEncoder.Model.Architecture.ParallelMLP import ParallelMLP
from FunctionEncoder.Model.Architecture.RepresentationEncoderDeepSets import RepresentationEncoderDeepSets


class FunctionEncoder(torch.nn.Module):
    """A function encoder learns basis functions/vectors over a Hilbert space.

    A function encoder learns basis functions/vectors over a Hilbert space. 
    Typically, this is a function space mapping to Euclidean vectors, but it can be any Hilbert space, IE probability distributions.
    This class has a general purpose algorithm which supports both deterministic and stochastic data.
    The only difference between them is the dataset used and the inner product definition.
    This class supports two methods for computing the coefficients of the basis function, also called a representation:
    1. "inner_product": It computes the inner product of the basis functions with the data via a Monte Carlo approximation.
    2. "least_squares": This method computes the least squares solution in terms of vector operations. This typically trains faster and better. 
    This class also supports the residuals method, which learns the average function in the dataset. The residuals/error of this approximation, 
    for each function in the space, is learned via a function encoder. This tends to improve performance when the average function is not f(x) = 0. 
    """

    def __init__(self,
                 input_size:tuple[int], 
                 output_size:tuple[int], 
                 data_type:str, 
                 n_basis:int=100, 
                 model_type:Union[str, type]="MLP",
                 model_kwargs:dict=dict(),
                 representation_mode:str="least_squares",
                 encoder_type:Union[str, type]="RepresentationEncoderDeepSets",
                 encoder_kwargs:dict=dict(),
                 use_residuals_method:bool=False,  
                 regularization_parameter:float=1.0, # if you normalize your data, this is usually good
                 gradient_accumulation:int=1, # default: no gradient accumulation
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs:dict={"lr":1e-3},
                 ):
        """ Initializes a function encoder.

        Args:
        input_size: tuple[int]: The size of the input space, e.g. (1,) for 1D input
        output_size: tuple[int]: The size of the output space, e.g. (1,) for 1D output
        data_type: str: "deterministic" or "stochastic". Determines which defintion of inner product is used.
        n_basis: int: Number of basis functions to use.
        model_type: str: The type of model to use for basis functions (and average function). See FunctionEncoder/Model/Architecture. Typically an MLP.
        model_kwargs: Union[dict, type(None)]: The kwargs to pass to the basis function model.
        representation_mode: str: "inner_product", "least_squares", or "encoder_network". Determines how to compute the coefficients (representation) of the basis functions.
        encoder_type: Union[str, type]: The type of model to use for the representation encoder if representation_mode is "encoder_network".
        encoder_kwargs: dict: The kwargs to pass to the representation encoder model.
        use_residuals_method: bool: Whether to use the residuals method. If True, uses an average function to predict the average of the data, and then learns the error with a function encoder.
        regularization_parameter: float: The regularization parameter for the least squares method, that encourages the basis functions to be unit length. 1 is usually good, but if your ys are very large, this may need to be increased. Only used if representation_mode="least_squares".
        gradient_accumulation: int: The number of batches to accumulate gradients over.
        optimizer: The optimizer class to use.
        optimizer_kwargs: Dict: Keyword arguments for the optimizer.
        """
        if model_type == "MLP":
            assert len(input_size) == 1, "MLP only supports 1D input"
        if model_type == "ParallelMLP":
            assert len(input_size) == 1, "ParallelMLP only supports 1D input"
        if model_type == "CNN":
            assert len(input_size) == 3, "CNN only supports 3D input"
        if isinstance(model_type, type):
            assert issubclass(model_type, BaseArchitecture), "model_type should be a subclass of BaseArchitecture. This just gives a way of predicting the number of parameters before init."
        assert len(input_size) in [1, 3], "Input must either be 1-Dimensional (euclidean vector) or 3-Dimensional (image)"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        assert representation_mode in ["inner_product", "least_squares", "encoder_network"], f"Unknown representation_mode: {representation_mode}"
        super(FunctionEncoder, self).__init__()
        
        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.representation_mode = representation_mode
        self.data_type = data_type
        
        # models and optimizers
        self.model = self._build_main_model(model_type, model_kwargs)
        self.average_function = self._build_main_model(model_type, model_kwargs, average_function=True) if use_residuals_method else None
        self.representation_encoder = None
        if self.representation_mode == "encoder_network":
            self.representation_encoder = self._build_encoder(encoder_type, encoder_kwargs)
        params = [*self.model.parameters()]
        if self.average_function is not None:
            params += [*self.average_function.parameters()]
        if self.representation_encoder is not None:
            params += [*self.representation_encoder.parameters()]
        self.opt = optimizer(params, **optimizer_kwargs) # usually ADAM with lr 1e-3

        # regulation only used for LS method
        self.regularization_parameter = regularization_parameter
        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons. 
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.encoder_type = encoder_type if self.representation_mode == "encoder_network" else None
        self.encoder_kwargs = encoder_kwargs if self.representation_mode == "encoder_network" else None

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = FunctionEncoder.predict_number_params(
            input_size=input_size,
            output_size=output_size,
            n_basis=n_basis,
            model_type=model_type,
            model_kwargs=model_kwargs,
            representation_mode=representation_mode,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            use_residuals_method=use_residuals_method
        )
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."



    def _build_main_model(self, 
               model_type:Union[str, type],
               model_kwargs:dict, 
               average_function:bool=False) -> torch.nn.Module:
        """Builds the basis function model or the average function model."""

        # if provided as a string, parse the string into a class
        if type(model_type) == str:
            if model_type == "MLP":
                return MLP(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=self.n_basis,
                           learn_basis_functions=not average_function,
                           **model_kwargs)
            if model_type == "ParallelMLP":
                return ParallelMLP(input_size=self.input_size,
                                   output_size=self.output_size,
                                   n_basis=self.n_basis,
                                   learn_basis_functions=not average_function,
                                   **model_kwargs)
            elif model_type == "Euclidean":
                return Euclidean(input_size=self.input_size,
                                 output_size=self.output_size,
                                 n_basis=self.n_basis,
                                 **model_kwargs)
            elif model_type == "CNN":
                return CNN(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=self.n_basis,
                           learn_basis_functions=not average_function,
                           **model_kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', or 'CNN'")
        else:  # otherwise, assume it is a class and directly instantiate it
            return model_type(input_size=self.input_size,
                              output_size=self.output_size,
                              n_basis=self.n_basis,
                              learn_basis_functions=not average_function,
                              **model_kwargs)

    def _build_encoder(self,
                       encoder_type:Union[str, type],
                       encoder_kwargs:dict) -> torch.nn.Module:
        """Builds the representation encoder model."""
        if type(encoder_type) == str:
            if encoder_type == "RepresentationEncoderDeepSets":
                return RepresentationEncoderDeepSets(input_size=self.input_size,
                                                     output_size=self.output_size,
                                                     n_basis=self.n_basis,
                                                     **encoder_kwargs)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
        elif issubclass(encoder_type, BaseArchitecture):
             return encoder_type(input_size=self.input_size,
                                 output_size=self.output_size,
                                 n_basis=self.n_basis,
                                 **encoder_kwargs)
        else:
            raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be a string or a BaseArchitecture subclass.")

    def compute_representation(self, 
                               example_xs:torch.tensor, 
                               example_ys:torch.tensor, 
                               **kwargs) -> Tuple[torch.tensor, Union[torch.tensor, None]]:
        """Computes the coefficients of the basis functions using the configured method.

        This method does the forward pass of the basis functions (and the average function if it exists) over the example data.
        Then it computes the coefficients of the basis functions via a Monte Carlo integration of the inner product with the example data.
        
        Args:
        example_xs: torch.tensor: The input data. Shape (n_example_datapoints, *input_size) or (n_functions, n_example_datapoints, *input_size)
        example_ys: torch.tensor: The output data. Shape (n_example_datapoints, *output_size) or (n_functions, n_example_datapoints, *output_size)
        kwargs: dict: Additional kwargs, primarily for the least squares method (e.g., 'lambd').

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis) or (n_basis,) if n_functions=1. 
        Union[torch.tensor, None]: The gram matrix if using least squares method. None otherwise.
        """
        
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) - len(self.input_size) == 1:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # optionally subtract average function if we are using residuals method
        # we dont want to backprop to the average function. So we block grads. 
        if self.average_function is not None:
            with torch.no_grad():
                example_y_hat_average = self.average_function.forward(example_xs)
                example_ys = example_ys - example_y_hat_average

        # compute representation based on the mode set during initialization
        Gs = self.model.forward(example_xs) # forward pass of the basis functions
        gram = None # Initialize gram to None

        if self.representation_mode == "inner_product":
            representation = self._compute_inner_product_representation(Gs, example_ys)
        elif self.representation_mode == "least_squares":
            representation, gram = self._compute_least_squares_representation(Gs, example_ys, **kwargs)
        elif self.representation_mode == "encoder_network":
            assert self.representation_encoder is not None, "Representation encoder is not built, but representation_mode='encoder_network'"
            # Encoder takes original examples (before subtracting average)
            # Need to handle the potential reshaping for the encoder input
            original_example_xs = example_xs
            original_example_ys = example_ys + example_y_hat_average if self.average_function is not None else example_ys
            representation = self.representation_encoder(original_example_xs, original_example_ys)
            # Gram matrix is not computed in this mode
        else:
            raise ValueError(f"Unknown representation_mode: {self.representation_mode}")

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def _deterministic_inner_product(self, 
                                     fs:torch.tensor, 
                                     gs:torch.tensor,) -> torch.tensor:
        """Approximates the L2 inner product between fs and gs using a Monte Carlo approximation.
        Latex: \langle f, g \rangle = \frac{1}{V}\int_X f(x)g(x) dx \approx \frac{1}{n} \sum_{i=1}^n f(x_i)g(x_i)
        Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
        Since scaling an inner product is still a valid inner product, this is still an inner product.
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self, 
                                  fs:torch.tensor, 
                                  gs:torch.tensor,) -> torch.tensor:
        """ Approximates the logit version of the inner product between continuous distributions. 
        Latex: \langle f, g \rangle = \int_X (f(x) - \Bar{f}(x) )(g(x) - \Bar{g}(x)) dx \approx \frac{1}{n} \sum_{i=1}^n (f(x_i) - \Bar{f}(x_i))(g(x_i) - \Bar{g}(x_i))
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _categorical_inner_product(self,
                                   fs:torch.tensor,
                                   gs:torch.tensor,) -> torch.tensor:
        """ Approximates the inner product between discrete conditional probability distributions.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=2, keepdim=True)
        mean_g = torch.mean(gs, dim=2, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self, 
                       fs:torch.tensor, 
                       gs:torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This passes the data to either the deterministic or stochastic inner product methods.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        elif self.data_type == "categorical":
            return self._categorical_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: '{self.data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

    def _norm(self, fs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs:torch.tensor, gs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the distance between fs and gs according to the chosen inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        gs: torch.tensor: The second set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        returns:
        torch.tensor: The distance between fs and gs.
        """
        return self._norm(fs - gs, squared=squared)

    def _compute_inner_product_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor) -> torch.tensor:
        """ Computes the coefficients via the inner product method.

        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        """
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"

        # take inner product with Gs, example_ys
        inner_products = self._inner_product(Gs, example_ys)
        return inner_products

    def _compute_least_squares_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor, 
                                              lambd:Union[float, type(None)]= None) -> Tuple[torch.tensor, torch.tensor]:
        """ Computes the coefficients via the least squares method.
        
        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)
        lambd: float: The regularization parameter. None by default. If None, scales with 1/n_datapoints.
        
        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        torch.tensor: The gram matrix. Shape (n_functions, n_basis, n_basis)
        """
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"
        assert lambd is None or lambd >= 0, f"Expected lambda to be non-negative or None, got {lambd}"

        # set lambd to decrease with more data
        if lambd is None:
            lambd = 1e-3 # emprically this does well. We need to investigate if there is an optimal value here.

        # compute gram
        gram = self._inner_product(Gs, Gs)
        gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

        # compute the matrix G^TF
        ip_representation = self._inner_product(Gs, example_ys)

        # Compute (G^TG)^-1 G^TF
        ls_representation = torch.einsum("fkl,fl->fk", gram_reg.inverse(), ip_representation) # this is just batch matrix multiplication
        return ls_representation, gram

    def predict(self, 
                query_xs:torch.tensor,
                representations:torch.tensor, 
                precomputed_average_ys:Union[torch.tensor, None]=None) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)
        
        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        assert query_xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {query_xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        Gs = self.model.forward(query_xs)
        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        
        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it
        if self.average_function:
            if precomputed_average_ys is not None:
                average_ys = precomputed_average_ys
            else:
                average_ys = self.average_function.forward(query_xs)
            y_hats = y_hats + average_ys
        return y_hats

    def predict_from_examples(self, 
                              example_xs:torch.tensor, 
                              example_ys:torch.tensor, 
                              query_xs:torch.tensor,
                              **kwargs):
        """ Predicts the output of the function encoder given the input data and the example data. Uses the average function if it exists.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_functions, n_example_datapoints, *input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_functions, n_example_datapoints, *output_size)
        query_xs: torch.tensor: The input data for which to predict outputs. Shape (n_functions, n_query_datapoints, *input_size)
        kwargs: dict: Additional kwargs passed to compute_representation (e.g., 'lambd' for least_squares).

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_query_datapoints, *output_size)
        """

        assert len(example_xs.shape) == 2 + len(self.input_size), f"Expected example_xs to have shape (f,d,*n), got {example_xs.shape}"
        assert len(example_ys.shape) == 2 + len(self.output_size), f"Expected example_ys to have shape (f,d,*m), got {example_ys.shape}"
        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"Expected example_xs to have shape (..., {self.input_size}), got {example_xs.shape[-1]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"Expected example_ys to have shape (..., {self.output_size}), got {example_ys.shape[-1]}"
        assert query_xs.shape[-len(self.input_size):] == self.input_size, f"Expected xs to have shape (..., {self.input_size}), got {query_xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == query_xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {query_xs.shape[0]}"

        representations, _ = self.compute_representation(example_xs, example_ys, **kwargs)
        y_hats = self.predict(query_xs, representations)
        return y_hats


    def estimate_L2_error(self, example_xs, example_ys):
        """ Estimates the L2 error of the function encoder on the example data. 
        This gives an idea if the example data lies in the span of the basis, or not.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_functions, n_example_datapoints, *input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_functions, n_example_datapoints, *output_size)
        
        Returns:
        torch.tensor: The estimated L2 error. Shape (n_functions,)
        """
        # Add check for representation mode
        if self.representation_mode != "least_squares":
            raise ValueError("estimate_L2_error is only supported for representation_mode='least_squares'")

        # Compute representation using least squares, regardless of the instance's default mode for this specific calculation
        representation, gram = self.compute_representation(example_xs, example_ys, method_override="least_squares") # Need to adjust compute_representation to allow override or call _compute_least_squares_representation directly

        # Let's call the internal method directly to avoid issues
        # Handle residuals if necessary
        current_example_ys = example_ys
        if self.average_function is not None:
             with torch.no_grad():
                example_y_hat_average = self.average_function.forward(example_xs)
                current_example_ys = example_ys - example_y_hat_average
        Gs = self.model.forward(example_xs)
        representation, gram = self._compute_least_squares_representation(Gs, current_example_ys) # Assuming default lambda is okay here

        # Ensure gram is not None (should be returned by LS)
        assert gram is not None, "Gram matrix should not be None when using least squares for L2 error estimation."

        # Calculate norms (need to handle potential batch dim in representation)
        # representation shape: (f, k), gram shape: (f, k, k)
        # Need element-wise batch matrix multiplication: bmm(representation.unsqueeze(1), bmm(gram, representation.unsqueeze(2))).squeeze()
        f_hat_norm_squared_diag = torch.einsum('fk,fkl,fl->f', representation, gram, representation) # More efficient way

        # Calculate norm of original function (or residuals)
        f_norm_squared = self._norm(current_example_ys, squared=True) # Shape (f,)

        # Clamp to avoid numerical issues with sqrt(negative)
        l2_distance_squared = torch.clamp(f_norm_squared - f_hat_norm_squared_diag, min=0.0)
        l2_distance = torch.sqrt(l2_distance_squared)
        return l2_distance



    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback:BaseCallback=None,
                    **kwargs):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training is now self.representation_mode
        assert self.representation_mode in ["inner_product", "least_squares", "encoder_network"], f"Unknown representation_mode: {self.representation_mode}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

            # train average function, if it exists
            if self.average_function is not None:
                # predict averages
                expected_yhats = self.average_function.forward(query_xs)

                # compute average function loss
                average_function_loss = self._distance(expected_yhats, query_ys, squared=True).mean()
                
                # we only train average function to fit data in general, so block backprop from the basis function loss
                expected_yhats = expected_yhats.detach()
            else:
                expected_yhats = None

            # approximate functions, compute error using the instance's mode
            representation, gram = self.compute_representation(example_xs, example_ys, **kwargs) # Pass kwargs for LS lambda if needed
            y_hats = self.predict(query_xs, representation, precomputed_average_ys=expected_yhats)
            prediction_loss = self._distance(y_hats, query_ys, squared=True).mean()

            # LS requires regularization since it does not care about the scale of basis
            # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
            # from going to infinity.
            norm_loss = 0.0
            if self.representation_mode == "least_squares":
                # Ensure gram is computed
                if gram is None:
                     # Recompute gram if not returned by compute_representation (e.g. if kwargs didn't trigger LS or mode changed)
                     # This case shouldn't happen if compute_representation is called correctly above for LS mode
                     # We might need Gs again if it wasn't stored
                     with torch.no_grad(): # Avoid recomputing gradients for Gs if possible
                         Gs = self.model.forward(example_xs)
                         gram = self._inner_product(Gs, Gs)

                assert gram is not None, "Gram matrix is None in least_squares mode during training."
                # Use torch.diagonal(gram, dim1=-2, dim2=-1) for safety with batch dim
                norm_loss = ((torch.diagonal(gram, dim1=-2, dim2=-1) - 1)**2).mean()

            # add loss components
            loss = prediction_loss
            if self.representation_mode == "least_squares":
                loss = loss + self.regularization_parameter * norm_loss
            if self.average_function is not None:
                loss = loss + average_function_loss
            
            # backprop with gradient clipping
            loss.backward()
            if (epoch+1) % self.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["input_size"] = self.input_size
        params["output_size"] = self.output_size
        params["n_basis"] = self.n_basis
        params["representation_mode"] = self.representation_mode
        params["model_type"] = self.model_type
        # Only add regularization parameter if relevant (LS mode)
        if self.representation_mode == "least_squares":
             params["regularization_parameter"] = self.regularization_parameter
        # Add model kwargs
        if self.model_kwargs: # Check if model_kwargs is not None or empty
            for k, v in self.model_kwargs.items():
                params[k] = v
        # Only add encoder kwargs if they exist (i.e., encoder_network mode)
        if self.encoder_kwargs: # Check if encoder_kwargs is not None or empty
            for k, v in self.encoder_kwargs.items():
                params[f"encoder_{k}"] = v
        params = {k: str(v) for k, v in params.items()}
        return params

    @staticmethod
    def predict_number_params(input_size:tuple[int],
                             output_size:tuple[int],
                             n_basis:int=100,
                             model_type:Union[str, type]="MLP",
                             model_kwargs:dict=dict(),
                             representation_mode:str = "least_squares",
                             encoder_type:Union[str, type] = "RepresentationEncoderDeepSets",
                             encoder_kwargs:dict = dict(),
                             use_residuals_method: bool = False,
                             *args, **kwargs):
        """ Predicts the number of parameters in the function encoder.
        Useful for ensuring all experiments use the same number of params"""
        n_params = 0
        # --- Parameters for Basis/Average Function Model ---
        if model_type == "MLP":
            n_params += MLP.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += MLP.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=False, **model_kwargs)
        elif model_type == "ParallelMLP":
            n_params += ParallelMLP.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += ParallelMLP.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "Euclidean":
            n_params += Euclidean.predict_number_params(output_size, n_basis)
            if use_residuals_method:
                n_params += Euclidean.predict_number_params(output_size, n_basis)
        elif model_type == "CNN":
            n_params += CNN.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += CNN.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif isinstance(model_type, type):
            n_params += model_type.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += model_type.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: '{model_type}'. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', or 'CNN'")

        # --- Parameters for Representation Encoder Model ---
        if representation_mode == "encoder_network":
            if encoder_type == "RepresentationEncoderDeepSets":
                n_params += RepresentationEncoderDeepSets.predict_number_params(input_size, output_size, n_basis, **encoder_kwargs)
            elif isinstance(encoder_type, type):
                 n_params += encoder_type.predict_number_params(input_size, output_size, n_basis, **encoder_kwargs)
            else:
                 raise ValueError(f"Unknown encoder type: '{encoder_type}'")

        return n_params

    def forward_basis_functions(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the basis functions. """
        return self.model.forward(xs)

    def forward_average_function(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the average function. """
        return self.average_function.forward(xs) if self.average_function is not None else None