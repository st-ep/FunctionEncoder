import torch
from torch.utils.tensorboard import SummaryWriter

# Assuming BaseCallback and TensorboardCallback are importable relative to this file
# Adjust the import path if BaseCallback is elsewhere
from .BaseCallback import BaseCallback
from .TensorboardCallback import TensorboardCallback


class OrthonormalityCallback(BaseCallback):
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
        self._last_logged_epoch = -1
        # --- Add lists to store history ---
        self.epoch_history = []
        self.error_history = []

    def on_training_start(self, trainer_locals):
        model = trainer_locals['self'] # Access the FunctionEncoder instance
        callback_list = trainer_locals['callback'] # Access the callback list/instance

        # Find the Tensorboard writer if not provided explicitly
        if self.tensorboard is None:
            # Check if the main callback is a ListCallback
            if hasattr(callback_list, 'callbacks'):
                 for cb in callback_list.callbacks:
                    if isinstance(cb, TensorboardCallback):
                        self.tensorboard = cb.tensorboard
                        break
            # Check if the main callback itself is a TensorboardCallback (less common)
            elif isinstance(callback_list, TensorboardCallback):
                 self.tensorboard = callback_list.tensorboard

        if self.tensorboard is None:
            print("Warning: OrthonormalityCallback requires a Tensorboard writer, "
                  "but none was provided or found.")
            return # Cannot proceed without tensorboard

        # Prepare integration points
        device = next(model.parameters()).device
        self._integration_points = torch.linspace(
            self.input_range[0], self.input_range[1], self.n_integration_points,
            device=device
        ).unsqueeze(-1) # Shape: (n_integration_points, 1)
        self._dx = (self.input_range[1] - self.input_range[0]) / self.n_integration_points
        self._last_logged_epoch = -1
        # --- Reset history lists ---
        self.epoch_history = []
        self.error_history = []

    def on_step(self, trainer_locals):
        model = trainer_locals['self']
        epoch = trainer_locals['epoch']
        # loss = trainer_locals['loss'] # Loss from the current step - not needed here

        if epoch % self.log_freq == 0 and epoch != self._last_logged_epoch and self._integration_points is not None:
            # --- Perform the calculation ---
            with torch.no_grad():
                # Evaluate basis functions at integration points
                # Output shape: (n_integration_points, 1, n_basis)
                basis_values = model.forward_basis_functions(self._integration_points)
                # Reshape to (n_integration_points, n_basis)
                basis_matrix = basis_values.squeeze(1)

                # Approximate Gram matrix G[i, j] = integral(phi_i(x) * phi_j(x) dx)
                # G approx dx * B^T @ B
                # Ensure basis_matrix is float for matmul
                gram_matrix = self._dx * (basis_matrix.float().T @ basis_matrix.float())

                # Target: Identity matrix
                identity_matrix = torch.eye(model.n_basis, device=gram_matrix.device, dtype=gram_matrix.dtype)

                # Calculate orthonormality error (Frobenius norm of difference)
                orthonormality_error = torch.norm(gram_matrix - identity_matrix, p='fro')

                # Log to TensorBoard (optional, can keep or remove)
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar(
                        'Orthonormality/Error',
                        orthonormality_error.item(),
                        epoch
                    )

                # --- Store history ---
                self.epoch_history.append(epoch)
                self.error_history.append(orthonormality_error.item())

            # --- Update the tracker ---
            self._last_logged_epoch = epoch

    # --- Add getter for history ---
    def get_history(self):
        """Returns the recorded epoch and orthonormality error history."""
        return self.epoch_history, self.error_history

    # Implement other methods like on_training_end if needed, otherwise inherit pass from BaseCallback
    # def on_training_end(self, trainer_locals):
    #     pass 