import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

class Callback:
    # ... existing Callback class ...

class ListCallback(Callback):
    # ... existing ListCallback class ...

class TensorboardCallback(Callback):
    # ... existing TensorboardCallback class ...

class MSECallback(Callback):
    # ... existing MSECallback class ...

class DistanceCallback(Callback):
    # ... existing DistanceCallback class ...

# --- New Callback ---
class OrthonormalityCallback(Callback):
    """
    Callback to measure and log the orthonormality of the basis functions during training.

    Computes the Frobenius norm of the difference between the Gram matrix
    of the basis functions (approximated via numerical integration) and the
    identity matrix. Logs the result to TensorBoard.
    """
    def __init__(self, input_range, n_integration_points=1000, log_freq=100, tensorboard_writer=None):
        """
        Args:
            input_range (tuple): The (min, max) range for numerical integration.
            n_integration_points (int): Number of points for numerical integration.
            log_freq (int): How often (in epochs) to compute and log the metric.
            tensorboard_writer (SummaryWriter, optional): TensorBoard writer instance.
                                                             If None, requires a TensorboardCallback
                                                             to be present in the ListCallback.
        """
        self.input_range = input_range
        self.n_integration_points = n_integration_points
        self.log_freq = log_freq
        self.tensorboard = tensorboard_writer
        self._integration_points = None
        self._dx = None

    def on_train_begin(self, trainer, model):
        # Find the Tensorboard writer if not provided explicitly
        if self.tensorboard is None:
            for cb in trainer.callback.callbacks: # Assuming ListCallback is used
                if isinstance(cb, TensorboardCallback):
                    self.tensorboard = cb.tensorboard
                    break
        if self.tensorboard is None:
            print("Warning: OrthonormalityCallback requires a Tensorboard writer, "
                  "but none was provided or found.")

        # Prepare integration points
        device = next(model.parameters()).device
        self._integration_points = torch.linspace(
            self.input_range[0], self.input_range[1], self.n_integration_points,
            device=device
        ).unsqueeze(-1) # Shape: (n_integration_points, 1)
        self._dx = (self.input_range[1] - self.input_range[0]) / self.n_integration_points

    def on_epoch_end(self, trainer, model, epoch, loss):
        if epoch % self.log_freq == 0 and self.tensorboard is not None:
            with torch.no_grad():
                # Evaluate basis functions at integration points
                # Output shape: (n_integration_points, 1, n_basis)
                basis_values = model.forward_basis_functions(self._integration_points)
                # Reshape to (n_integration_points, n_basis)
                basis_matrix = basis_values.squeeze(1)

                # Approximate Gram matrix G[i, j] = integral(phi_i(x) * phi_j(x) dx)
                # G approx dx * B^T @ B
                gram_matrix = self._dx * (basis_matrix.T @ basis_matrix)

                # Target: Identity matrix
                identity_matrix = torch.eye(model.n_basis, device=gram_matrix.device)

                # Calculate orthonormality error (Frobenius norm of difference)
                orthonormality_error = torch.norm(gram_matrix - identity_matrix, p='fro')

                # Log to TensorBoard
                self.tensorboard.add_scalar(
                    'Orthonormality/Error',
                    orthonormality_error.item(),
                    epoch
                )

# ... rest of the file ... 