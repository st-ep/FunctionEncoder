from datetime import datetime

import matplotlib.pyplot as plt
import torch
import time

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback, OrthonormalityCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--representation_mode", type=str, default="least_squares",
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation *during training* or if not loading.")
parser.add_argument("--eval_mode", type=str, default=None,
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation *during evaluation* (overrides training mode if specified).")
parser.add_argument("--epochs", type=int, default=150000)
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
eval_mode = args.eval_mode if args.eval_mode is not None else representation_mode
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/quadratic_example/{representation_mode}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(seed)

# create a dataset
if residuals:
    a_range = (0, 3/50) # this makes the true average function non-zero
else:
    a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range)

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
                            model_type=arch,
                            representation_mode=representation_mode,
                            # Pass kwargs only if using encoder network
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    ortho_cb = OrthonormalityCallback(input_range=input_range, tensorboard_writer=cb1.tensorboard, log_freq=100)
    callback = ListCallback([cb1, cb2, ortho_cb])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")

    # --- Plot Orthonormality History ---
    ortho_epochs, ortho_errors = ortho_cb.get_history()
    if ortho_epochs: # Check if history is not empty
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
        plt.clf() # Clear the figure for the next plots
    else:
        print("DEBUG: No orthonormality history recorded, skipping plot.")
    # --- End Orthonormality Plot ---

else:
    # --- Load Model Section ---
    print(f"INFO: Loading model from {load_path}")
    print(f"INFO: Model was trained with representation_mode='{representation_mode}'")
    print(f"INFO: Evaluating with representation_mode='{eval_mode}'")

    # Define default encoder kwargs for loading
    # These should match the settings used during training if loading an encoder network model
    # We need these even if evaluating with a different mode, to ensure the state_dict loads correctly.
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean", # Or "attention"
        use_layer_norm=False # Set to True if the loaded model used LayerNorm
    )

    # Instantiate the model with the configuration it was *trained* with
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            representation_mode=representation_mode, # Use the original training mode here
                            # Pass kwargs if the *trained* model used encoder network
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(),
                            use_residuals_method=residuals).to(device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))
    print(f"INFO: Model state_dict loaded successfully.")

    # <<<--- SWITCH REPRESENTATION MODE FOR EVALUATION --- >>>
    if args.eval_mode is not None and args.eval_mode != representation_mode:
        print(f"INFO: Switching model's representation mode to '{args.eval_mode}' for evaluation.")
        model.representation_mode = args.eval_mode
    # <<<-------------------------------------------------- >>>

    print("INFO: Skipping orthonormality plot as model was loaded from path.")
    # --- End Load Model Section ---


# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats = model.predict_from_examples(example_xs, example_ys, query_xs)
    query_xs, indicies = torch.sort(query_xs, dim=-2)
    query_ys = query_ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(query_xs[i].cpu(), query_ys[i].cpu(), label="True")
        ax.plot(query_xs[i].cpu(), y_hats[i].cpu(), label=f"Pred ({eval_mode})")
        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        y_min, y_max = query_ys[i].min().item(), query_ys[i].max().item()
        ax.set_ylim(y_min, y_max)

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
        ax.plot(basis_plot_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if residuals:
        avg_function = model.average_function.forward(basis_plot_xs)
        ax.plot(basis_plot_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.tight_layout()
    basis_path = f"{logdir}/basis.png"
    print(f"DEBUG: Attempting to save basis plot to: {basis_path}")
    plt.savefig(basis_path)
    print(f"DEBUG: Basis plot save command executed.")
    # plt.clf() # Optional: comment out the last clf just in case

    # Test inference speed
    device = next(model.parameters()).device
    # Use the example data generated for the plots
    test_example_xs = example_xs.to(device)
    test_example_ys = example_ys.to(device)

    # --- Time Representation Computation ---
    # This part now times the specific mode set for evaluation
    torch.cuda.synchronize() # Ensure previous GPU work is done (if using GPU)
    start_time = time.perf_counter()
    with torch.no_grad():
        # compute_representation uses the potentially switched model.representation_mode
        computed_reps, _ = model.compute_representation(test_example_xs, test_example_ys)
    torch.cuda.synchronize() # Ensure GPU work is done
    end_time = time.perf_counter()
    rep_time = end_time - start_time
    # Print the mode actually used for timing
    print(f"{eval_mode.upper()} Representation Time: {rep_time:.6f}s")

    # Test prediction accuracy using the correct query data
    # Use the query_xs and query_ys generated earlier for the plots
    test_query_xs_acc = query_xs.to(device) # Shape (n_plots, n_query_points, 1)
    test_query_ys_acc = query_ys.to(device) # Shape (n_plots, n_query_points, 1)

    with torch.no_grad():
        # Predict using the same example data as the timing test
        # predict_from_examples also uses the potentially switched model.representation_mode
        y_hats_acc = model.predict_from_examples(test_example_xs, test_example_ys, test_query_xs_acc)

        # Calculate MSE against the *correct* target values
        mse_acc = torch.mean((y_hats_acc - test_query_ys_acc)**2)

    # Print the MSE for the mode actually used for evaluation
    print(f"{eval_mode.upper()} Prediction MSE on test functions: {mse_acc.item():.6f}")


