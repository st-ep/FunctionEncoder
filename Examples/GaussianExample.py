import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np # Import numpy for calculations

from FunctionEncoder import FunctionEncoder, GaussianDataset, TensorboardCallback, ListCallback, DistanceCallback

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
device = "cuda" if torch.cuda.is_available() else "cpu"
representation_mode = args.representation_mode
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/gaussian_example/{representation_mode}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# create dataset
dataset = GaussianDataset()

if load_path is None:
    # You can customize these hyperparameters
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean",
        use_layer_norm=True
    )

    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode,
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
    # load the model
    # Make sure these match the ones used during training if loading a model
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean",
        use_layer_norm=True
    )
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode,
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
    # Move example data to device
    example_xs = example_xs.to(device)
    example_ys = example_ys.to(device)

    # Expand grid to match number of functions in example_xs
    n_functions = example_xs.shape[0]
    xs_grid_axis = torch.arange(-1, 1, 0.02, device=device) # Grid axis
    grid_dim = len(xs_grid_axis) # Store the dimension size (e.g., 100)
    xs = torch.stack(torch.meshgrid(xs_grid_axis, xs_grid_axis, indexing="ij"), dim=-1).reshape(-1, 2)
    xs = xs.unsqueeze(0).expand(n_functions, -1, -1) # Shape: (n_functions, n_grid_points, 2)

    # Remove the 'method' argument from predict_from_examples
    logits = model.predict_from_examples(example_xs, example_ys, xs)

    # The output 'logits' are not normalized probabilities yet.
    # We need to exponentiate and normalize.
    # Normalization requires integration (approximated by summation over the grid)
    delta_area = (xs_grid_axis[1] - xs_grid_axis[0])**2 # Area of each grid cell based on original axis
    probs = torch.exp(logits) # Convert logits to unnormalized probabilities
    # Sum over grid points (dim=1) and multiply by cell area for approximate integral
    integral_approx = torch.sum(probs, dim=1, keepdim=True) * delta_area
    pdf = probs / integral_approx # Normalize to make it a PDF

    pdf = pdf.to("cpu").numpy()
    pdf = pdf.reshape(n_functions, grid_dim, grid_dim)
    xs_plot = xs_grid_axis.cpu().numpy() # Use the original grid axis for plotting coordinates

    std_devs = info["std_devs"]
    kl_divergences = [] # List to store KL divergences

    # Create grid coordinates for true PDF calculation
    xx, yy = np.meshgrid(xs_plot, xs_plot)
    grid_coords_sq = xx**2 + yy**2
    epsilon = 1e-10 # Small value for numerical stability

    for i in range(min(9, n_functions)):
        ax = axes[i]
        sigma = std_devs[i].item()
        q_pdf = pdf[i] # Predicted PDF

        # Calculate true Gaussian PDF on the grid
        p_pdf = (1 / (2 * np.pi * sigma**2)) * np.exp(-grid_coords_sq / (2 * sigma**2))

        # Normalize both PDFs over the discrete grid to sum to 1
        p_norm = p_pdf / np.sum(p_pdf)
        q_norm = q_pdf / np.sum(q_pdf)

        # Calculate KL Divergence D_KL(P || Q)
        kl_div = np.sum(p_norm * (np.log(p_norm + epsilon) - np.log(q_norm + epsilon)))
        kl_divergences.append(kl_div)

        contour = ax.contourf(xs_plot, xs_plot, q_pdf, levels=100, cmap="Reds") # Plot predicted PDF
        ax.scatter(example_xs[i, :, 0].cpu(), example_xs[i, :, 1].cpu(), color="black", s=1, alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # Add KL divergence to the title
        ax.set_title(f"Std Dev: {sigma:.2f}\nKL Div: {kl_div:.3f}")

    # color bar - use the last contour plot created
    if 'contour' in locals():
        cax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(contour, cax=cax, orientation="vertical", fraction=0.1)
    else:
         print("Warning: No contours were plotted, skipping colorbar.")

    plt.tight_layout()
    plt.savefig(f"{logdir}/gaussians.png")
    print(f"Saved plot to {logdir}/gaussians.png")
    plt.clf() # Clear the figure for the next plot

    # --- Plot Basis Functions ---
    n_basis_plots = min(9, n_basis) # Plot up to 9 basis functions
    n_basis_cols = 3
    n_basis_rows = (n_basis_plots + n_basis_cols - 1) // n_basis_cols
    # Increase vertical space slightly with figsize
    fig_basis, axes_basis = plt.subplots(n_basis_rows, n_basis_cols, figsize=(n_basis_cols * 4, n_basis_rows * 4.2), squeeze=False)
    fig_basis.suptitle("First Few Basis Functions", fontsize=16)

    # Create a grid for evaluating basis functions
    basis_grid_axis = torch.arange(-1, 1, 0.05, device=device) # Coarser grid for basis viz
    basis_grid_dim = len(basis_grid_axis)
    basis_xs = torch.stack(torch.meshgrid(basis_grid_axis, basis_grid_axis, indexing="ij"), dim=-1).reshape(-1, 2)
    # Shape: (n_grid_points, 2)

    # Evaluate basis functions on the grid
    basis_values = model.forward_basis_functions(basis_xs) # Shape: (n_grid_points, output_size=1, n_basis)
    basis_values = basis_values.squeeze(1) # Shape: (n_grid_points, n_basis)
    basis_values = basis_values.reshape(basis_grid_dim, basis_grid_dim, n_basis) # Shape: (grid_dim, grid_dim, n_basis)
    basis_values = basis_values.cpu().numpy()
    basis_xs_plot = basis_grid_axis.cpu().numpy()

    last_contour = None
    for i in range(n_basis_plots):
        row, col = i // n_basis_cols, i % n_basis_cols
        ax = axes_basis[row, col]
        basis_i_values = basis_values[:, :, i]
        # Use a consistent color scale if desired, otherwise let each plot auto-scale
        # vmin, vmax = np.min(basis_values), np.max(basis_values) # Uncomment for consistent scale
        # contour = ax.contourf(basis_xs_plot, basis_xs_plot, basis_i_values, levels=50, cmap="viridis", vmin=vmin, vmax=vmax)
        contour = ax.contourf(basis_xs_plot, basis_xs_plot, basis_i_values, levels=50, cmap="viridis")
        ax.set_title(f"Basis {i}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        last_contour = contour # Keep track of the last contour for the colorbar

    # Remove empty subplots if n_basis_plots is not a multiple of n_basis_cols
    for i in range(n_basis_plots, n_basis_rows * n_basis_cols):
        row, col = i // n_basis_cols, i % n_basis_cols
        fig_basis.delaxes(axes_basis[row, col])

    # Adjust layout *before* adding the colorbar to make space
    # Increase hspace and wspace for better separation
    fig_basis.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Leave space on right for colorbar, top for suptitle
    fig_basis.subplots_adjust(hspace=0.4, wspace=0.3) # Add more explicit spacing

    # Add a colorbar in a dedicated space to the right
    if last_contour:
        # Position the colorbar axis: [left, bottom, width, height] in figure coordinates
        cbar_ax = fig_basis.add_axes([0.92, 0.15, 0.03, 0.7]) # Adjust these values as needed
        fig_basis.colorbar(last_contour, cax=cbar_ax)


    basis_plot_path = f"{logdir}/basis_functions.png"
    plt.savefig(basis_plot_path)
    print(f"Saved basis function plot to {basis_plot_path}")
    plt.clf()
    # --- End Basis Function Plot ---


    # Print average KL divergence
    if kl_divergences:
        avg_kl = np.mean(kl_divergences)
        print(f"Average KL Divergence (P_true || Q_pred) over {len(kl_divergences)} examples: {avg_kl:.4f}")
