import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import GaussianDonutDataset, FunctionEncoder, DistanceCallback, TensorboardCallback, ListCallback

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--representation_mode", type=str, default="least_squares",
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation.")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda"
representation_mode = args.representation_mode
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/donut_example/{representation_mode}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)


# create dataset
dataset = GaussianDonutDataset(noise=0.1)

if load_path is None:
    # Define default encoder kwargs (can be customized)
    # These are only used if representation_mode == "encoder_network"
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean", # Or "attention"
        use_layer_norm=False # Set to True to enable LayerNorm
    )

    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode,
                            # Pass kwargs only if using encoder network
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # Define default encoder kwargs for loading as well
    # These should match the settings used during training if loading an encoder network model
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean", # Or "attention"
        use_layer_norm=False # Set to True if the loaded model used LayerNorm
    )
    # Update FunctionEncoder instantiation for loading
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode,
                            # Pass kwargs only if using encoder network
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))



# plot
with torch.no_grad():
    n_cols, n_rows = 3, 3
    fig = plt.figure(figsize=(n_cols * 4 + 1, n_rows * 3.8))
    gs = plt.GridSpec(n_rows, n_cols + 1,  width_ratios=[4, 4, 4, 1])
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols], aspect='equal') for i in range(n_cols * n_rows)]

    example_xs, example_ys, _, _, info = dataset.sample()

    # compute pdf over full space
    # compute pdf at grid points and plot using plt
    grid = torch.arange(-1, 1, 0.02, device=device)
    xs = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 2).expand(10, -1, -1)

    # compute pdf
    logits = model.predict_from_examples(example_xs, example_ys, xs)
    e_logits = torch.exp(logits)
    sums = torch.mean(e_logits, dim=1, keepdim=True) * dataset.volume
    pdf = e_logits / sums
    grid = grid.to("cpu").numpy()
    pdf = pdf.to("cpu").numpy()
    pdf = pdf.reshape(10, len(grid), len(grid))



    radii = info["radii"]
    for i in range(9):
        ax = axes[i ]
        ax.contourf(grid, grid, pdf[i], levels=100, cmap="Reds", )
        ax.scatter(example_xs[i, :example_xs.shape[1]//2, 0].cpu(), example_xs[i, :example_xs.shape[1]//2, 1].cpu(), color="black", s=1, alpha=0.5)
        circle = plt.Circle((0, 0), radii[i].cpu(), color='b', fill=False)
        ax.add_artist(circle)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # color bar
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)

    plt.tight_layout()
    plt.savefig(f"{logdir}/donuts.png")
