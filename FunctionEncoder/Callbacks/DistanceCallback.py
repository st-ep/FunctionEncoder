from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class DistanceCallback(BaseCallback):

    def __init__(self,
                 testing_dataset:BaseDataset,
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix="test",
                 log_freq: int = 100,
                 ):
        """ Constructor for MSECallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(DistanceCallback, self).__init__()
        self.testing_dataset = testing_dataset
        if logdir is not None:
            self.tensorboard = SummaryWriter(logdir)
        else:
            self.tensorboard = tensorboard
        self.prefix = prefix
        self.total_epochs = 0
        self.log_freq = log_freq

    def on_training_start(self, locals: dict) -> None:
        if self.total_epochs == 0: # logs loss before any updates.
            self.on_step(locals)

    def on_step(self, locals_dict: dict):
        epoch = locals_dict.get('epoch', self.total_epochs)

        if epoch % self.log_freq == 0:
            function_encoder = locals_dict['self'] # Get the FunctionEncoder instance
            # Get the device from the model's parameters
            try:
                device = next(function_encoder.parameters()).device
            except StopIteration:
                # Handle case where model has no parameters (unlikely for FunctionEncoder)
                print("Warning: FunctionEncoder has no parameters. Assuming CPU.")
                device = torch.device("cpu")

            with torch.no_grad():
                # Sample data
                example_xs, example_ys, query_xs, query_ys, _ = self.testing_dataset.sample()

                # Move sampled data to the function_encoder's device
                example_xs = example_xs.to(device)
                example_ys = example_ys.to(device)
                query_xs = query_xs.to(device)
                query_ys = query_ys.to(device) # Also move query_ys for consistency

                # Predict using the model and data on the same device
                y_hats = function_encoder.predict_from_examples(example_xs, example_ys, query_xs)

                # compute distance (using MSE as an example, matching previous logic)
                # Ensure the metric calculation handles potential shape differences if needed
                # and uses tensors on the correct device.
                if hasattr(function_encoder, '_distance'):
                    # Use the model's internal distance if available
                    distance = function_encoder._distance(y_hats, query_ys, squared=True).mean()
                else:
                    # Default to MSE loss if _distance is not available
                    distance = torch.nn.functional.mse_loss(y_hats, query_ys)

            # log results
            self.tensorboard.add_scalar(f"{self.prefix}/mean_distance_squared", distance, epoch)
            self.total_epochs += 1

