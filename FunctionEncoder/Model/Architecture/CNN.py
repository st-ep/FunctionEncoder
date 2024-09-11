import torch

from FunctionEncoder.Model.Architecture.MLP import get_activation


class CNN(torch.nn.Module):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size, n_layers):
        channels, height, width = input_size
        output_size = output_size[0] * n_basis
        kernel_size = 3

        n_params = 0
        # first conv2d
        n_params += (kernel_size*kernel_size)*(channels)*(2*channels) + 2*channels
        # second conv2d
        n_params += (kernel_size*kernel_size)*(2*channels)*(4*channels) + 4*channels
        # third conv2d
        n_params += (kernel_size*kernel_size)*(4*channels)*(8*channels) + 8*channels
        flatten_size = 8*channels * (height//2**3) * (width//2**3)
        # first linear
        n_params += flatten_size * hidden_size + hidden_size
        # hidden layers
        n_params += (hidden_size+1) * hidden_size * (n_layers - 2)
        # last linear
        n_params += (hidden_size+1) * output_size

        return n_params

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=256,
                 n_layers:int=3,
                 activation:str="relu"):
        super(CNN, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        assert len(input_size) == 3, "input_size must be a tuple of length 3, CHW"
        assert input_size[-3] <= 4, f"input_size[-3] must be <= 4 for RGB (and maybe D). Got {input_size[-3]}. Image order should be CHW. "
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis


        # get inputs
        output_size = output_size[0] * n_basis

        # image params
        channels = input_size[0]
        height = input_size[1]
        width = input_size[2]

        # build net
        layers = []

        # CNN part of net
        layers.append(torch.nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(torch.nn.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(torch.nn.Conv2d(4*channels, 8*channels, kernel_size=3, padding=1))
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(torch.nn.Flatten())
        flatten_size = 8*channels * (height//2**3) * (width//2**3)

        # MLP part of net
        layers.append(torch.nn.Linear(flatten_size, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

        assert sum([p.numel() for p in self.parameters()]) == self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers), f"Number of parameters is not as expected, got {sum([p.numel() for p in self.parameters()])} but expected {self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers)}"



    def forward(self, x):
        assert len(x.shape) >= 3, f"Expected at least 3 dimensions, got {len(x.shape)} with values {x.shape}"
        assert x.shape[-1] == self.input_size[-1], f"Expected {self.input_size[-1]} channels, got {x.shape[-1]}"
        assert x.shape[-2] == self.input_size[-2], f"Expected {self.input_size[-2]} height, got {x.shape[-2]}"
        assert x.shape[-3] == self.input_size[-3], f"Expected {self.input_size[-3]} width, got {x.shape[-3]}"


        reshape = None
        if len(x.shape) == 3:
            reshape = 1
            x = x.reshape(1, 1, *x.shape)
        if len(x.shape) == 4:
            reshape = 2
            x = x.reshape(1, *x.shape)

        # this is the main part of this function. The rest is just error handling
        # flatten the batch dims. Torch only supports 1 batch dim for images
        outs = self.model(x.reshape(-1, *x.shape[2:]))
        outs = outs.reshape(x.shape[0], x.shape[1], *outs.shape[1:])

        # reshape output dims
        if self.n_basis > 1:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs


