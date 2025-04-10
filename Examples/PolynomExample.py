from datetime import datetime

import matplotlib.pyplot as plt
import torch
import time

# Import PolynomDataset instead of QuadraticDataset
from FunctionEncoder import PolynomDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback, OrthonormalityCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
# Increased default n_basis slightly for higher order polynomial
parser.add_argument("--n_basis", type=int, default=15)
parser.add_argument("--representation_mode", type=str, default="least_squares",
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation.")
parser.add_argument("--epochs", type=int, default=300000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
representation_mode = args.representation_mode
seed = args.seed
load_path = args.load_path
residuals = args.residuals
# Update log directory name
if load_path is None:
    logdir = f"logs/polynom_example/{representation_mode}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(seed)

# create a dataset using PolynomDataset
# Using smaller ranges for coefficients due to higher powers
a_range = (-0.1, 0.1)
b_range = (-0.1, 0.1)
c_range = (-0.1, 0.1)
d_range = (-0.5, 0.5)
e_range = (-0.5, 0.5)
f_range = (-1.0, 1.0)
input_range = (-5, 5) # Adjusted input range slightly

# Instantiate PolynomDataset
dataset = PolynomDataset(a_range=a_range, b_range=b_range, c_range=c_range,
                         d_range=d_range, e_range=e_range, f_range=f_range,
                         input_range=input_range, device=device)

if load_path is None:
    # Define default encoder kwargs (can be customized)
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
                            model_type=arch,
                            representation_mode=representation_mode,
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir)
    # DistanceCallback uses the PolynomDataset instance
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard)
    ortho_cb = OrthonormalityCallback(input_range=input_range, tensorboard_writer=cb1.tensorboard, log_freq=100)
    callback = ListCallback([cb1, cb2, ortho_cb])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")

    # --- Plot Orthonormality History ---
    ortho_epochs, ortho_errors = ortho_cb.get_history()
    if ortho_epochs:
        fig_ortho, ax_ortho = plt.subplots(1, 1, figsize=(10, 6))
        ax_ortho.plot(ortho_epochs, ortho_errors, marker='o', linestyle='-')
        ax_ortho.set_xlabel("Epoch")
        ax_ortho.set_ylabel("Orthonormality Error (Frobenius Norm)")
        ax_ortho.set_title("Basis Orthonormality Error during Training")
        ax_ortho.grid(True)
        plt.tight_layout()
        ortho_plot_path = f"{logdir}/orthonormality_plot.png"
        print(f"DEBUG: Attempting to save orthonormality plot to: {ortho_plot_path}")
        plt.savefig(ortho_plot_path)
        print(f"DEBUG: Orthonormality plot save command executed.")
        plt.clf()
    else:
        print("DEBUG: No orthonormality history recorded, skipping plot.")
    # --- End Orthonormality Plot ---

else:
    # Define default encoder kwargs for loading
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean",
        use_layer_norm=False
    )

    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            representation_mode=representation_mode,
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))
    print("INFO: Skipping orthonormality plot as model was loaded from path.")

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100 # Number of examples used to compute representation
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    # Use only n_examples points for representation calculation if needed
    example_xs_rep = example_xs[:, :n_examples, :]
    example_ys_rep = example_ys[:, :n_examples, :]

    # Predict on the full query set
    y_hats = model.predict_from_examples(example_xs_rep, example_ys_rep, query_xs)

    # Sort query points for smoother plotting
    query_xs_sorted, indicies = torch.sort(query_xs, dim=-2)
    query_ys_sorted = query_ys.gather(dim=-2, index=indicies)
    y_hats_sorted = y_hats.gather(dim=-2, index=indicies)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12)) # Adjusted figsize slightly
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(query_xs_sorted[i].cpu(), query_ys_sorted[i].cpu(), label="True")
        ax.plot(query_xs_sorted[i].cpu(), y_hats_sorted[i].cpu(), label=f"Pred ({representation_mode})", linestyle='--')
        # Scatter plot the example points used for representation
        ax.scatter(example_xs_rep[i].cpu(), example_ys_rep[i].cpu(), color='red', s=10, label=f'{n_examples} Examples', alpha=0.6, zorder=5)

        if i == n_plots - 1:
            ax.legend()

        # Update title formatting for 5th order polynomial
        title = (f"${info['As'][i].item():.2f}x^5 + {info['Bs'][i].item():.2f}x^4 + {info['Cs'][i].item():.2f}x^3 + "
                 f"{info['Ds'][i].item():.2f}x^2 + {info['Es'][i].item():.2f}x + {info['Fs'][i].item():.2f}$")
        ax.set_title(title, fontsize=10) # Slightly smaller font for longer title
        y_min, y_max = query_ys_sorted[i].min().item(), query_ys_sorted[i].max().item()
        padding = (y_max - y_min) * 0.1 # Add padding to y-axis
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True)

    plt.tight_layout()
    plot_path = f"{logdir}/plot.png"
    print(f"DEBUG: Attempting to save main plot to: {plot_path}")
    plt.savefig(plot_path)
    print(f"DEBUG: Main plot save command executed.")
    plt.clf()

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    basis_plot_xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    basis = model.forward_basis_functions(basis_plot_xs)
    for i in range(n_basis):
        ax.plot(basis_plot_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black", alpha=0.7)
    if residuals:
        avg_function = model.average_function.forward(basis_plot_xs)
        ax.plot(basis_plot_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue", linewidth=2.5, label="Avg Function")
        ax.legend()

    ax.set_title(f"Learned Basis Functions (n={n_basis})")
    ax.set_xlabel("x")
    ax.set_ylabel("Basis Value")
    ax.grid(True)
    plt.tight_layout()
    basis_path = f"{logdir}/basis.png"
    print(f"DEBUG: Attempting to save basis plot to: {basis_path}")
    plt.savefig(basis_path)
    print(f"DEBUG: Basis plot save command executed.")
    plt.clf()

    # Test inference speed
    device = next(model.parameters()).device
    # Use the example data generated for the plots
    test_example_xs = example_xs_rep.to(device)
    test_example_ys = example_ys_rep.to(device)

    # --- Time Representation Computation ---
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        computed_reps, _ = model.compute_representation(test_example_xs, test_example_ys)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    rep_time = end_time - start_time
    print(f"{representation_mode.upper()} Representation Time ({n_examples} examples): {rep_time:.6f}s")

    # Test prediction accuracy using the correct query data
    test_query_xs_acc = query_xs.to(device) # Use the full query set
    test_query_ys_acc = query_ys.to(device) # Use the full query set

    with torch.no_grad():
        # Predict using the same example data as the timing test
        y_hats_acc = model.predict_from_examples(test_example_xs, test_example_ys, test_query_xs_acc)

        # Calculate MSE against the *correct* target values
        mse_acc = torch.mean((y_hats_acc - test_query_ys_acc)**2)

    print(f"{representation_mode.upper()} Prediction MSE on test functions: {mse_acc.item():.6f}")
