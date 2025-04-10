import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from FunctionEncoder import GaussianDonutDataset, FunctionEncoder, DistanceCallback, TensorboardCallback, ListCallback

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--representation_mode", type=str, default="least_squares",
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation.")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cpu"
representation_mode = args.representation_mode
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/conditional_donut_example/{representation_mode}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)


# create dataset
# Explicitly pass the desired device to the dataset constructor
dataset = GaussianDonutDataset(noise=0.1, device=device)

if load_path is None:
    # Define default encoder kwargs (can be customized)
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean"
    )

    # create the model for the marginal distribution
    marginal_distribution_model = FunctionEncoder(input_size=dataset.input_size,
                                                    output_size=dataset.output_size,
                                                    data_type=dataset.data_type,
                                                    n_basis=n_basis,
                                                    representation_mode=representation_mode,
                                                    encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                                                    use_residuals_method=residuals).to(device)
    print('Marginal Model - Number of parameters:', sum(p.numel() for p in marginal_distribution_model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    marginal_distribution_model.train_model(dataset, epochs=epochs, callback=callback)
    torch.save(marginal_distribution_model.state_dict(), f"{logdir}/marginal_distribution_model.pth")

    # now train a normal neural network to predict marginal distributions from radius
    input_size = 1
    output_size = args.n_basis
    conditional_model = torch.nn.Sequential(torch.nn.Linear(input_size, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, output_size)).to(device)
    optimizer = torch.optim.Adam(conditional_model.parameters(), lr=1e-3)
    dataset.n_functions_per_sample = 100

    # training loop
    for epoch in trange(epochs):
        with torch.no_grad():
            # get outputs (distribution as represented by coefficients)
            example_xs, example_ys, _, _, info = dataset.sample()
            representation, _ = marginal_distribution_model.compute_representation(example_xs, example_ys)

            # get inputs ( radii)
            radii = info["radii"]

        optimizer.zero_grad()
        pred = conditional_model(radii)
        assert pred.shape == representation.shape, f"{pred.shape} != {representation.shape}"
        loss = torch.nn.MSELoss()(pred, representation)
        loss.backward()
        optimizer.step()
        cb1.tensorboard.add_scalar("conditional_loss", loss.item(), epoch)
    
    torch.save(conditional_model.state_dict(), f"{logdir}/conditional_model.pth")

else:
    # Define default encoder kwargs for loading as well
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean"
    )
    # load the model
    marginal_distribution_model = FunctionEncoder(input_size=dataset.input_size,
                                                output_size=dataset.output_size,
                                                data_type=dataset.data_type,
                                                n_basis=n_basis,
                                                representation_mode=representation_mode,
                                                encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                                                use_residuals_method=residuals).to(device)
    input_size = 1
    output_size = args.n_basis
    marginal_distribution_model.load_state_dict(torch.load(f"{logdir}/marginal_distribution_model.pth"))
    conditional_model = torch.nn.Sequential(torch.nn.Linear(input_size, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, output_size)).to(device)
    conditional_model.load_state_dict(torch.load(f"{logdir}/conditional_model.pth"))




# plot
with torch.no_grad():
    n_cols, n_rows = 3, 3
    fig = plt.figure(figsize=(n_cols * 4 + 1, n_rows * 3.8))
    gs = plt.GridSpec(n_rows, n_cols + 1,  width_ratios=[4, 4, 4, 1])
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols], aspect='equal') for i in range(n_cols * n_rows)]

    # _, _, _, _, info = dataset.sample() # Don't need this sample anymore for radii

    # Generate 100 radii specifically for plotting
    plot_radii = torch.rand((100, 1), device=device) * dataset.radius # Use dataset's radius attribute

    # compute pdf over full space
    # compute pdf at grid points and plot using plt
    grid = torch.arange(-1, 1, 0.02, device=device)
    xs = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 2).expand(100, -1, -1)

    # compute pdf using the 100 generated radii
    representation = conditional_model(plot_radii) # Pass the 100 radii
    logits = marginal_distribution_model.predict(xs, representation)
    e_logits = torch.exp(logits)
    sums = torch.mean(e_logits, dim=1, keepdim=True) * dataset.volume
    pdf = e_logits / sums
    grid = grid.to("cpu").numpy()
    pdf = pdf.to("cpu").numpy()
    pdf = pdf.reshape(100, len(grid), len(grid))


    # Use the generated plot_radii for plotting circles
    # radii = info["radii"] # Remove this line
    for i in range(n_cols * n_rows): # Use n_cols * n_rows for the loop range (9 plots)
        ax = axes[i]
        ax.contourf(grid, grid, pdf[i], levels=100, cmap="Reds", )
        # ax.scatter(example_xs[i, :example_xs.shape[1]//2, 0].cpu(), example_xs[i, :example_xs.shape[1]//2, 1].cpu(), color="black", s=1, alpha=0.5) # Example points not available here
        # Use plot_radii[i] for the circle
        circle = plt.Circle((0, 0), plot_radii[i].item(), color='b', fill=False)
        ax.add_artist(circle)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # color bar
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)

    plt.tight_layout()
    plt.savefig(f"{logdir}/donuts.png")
