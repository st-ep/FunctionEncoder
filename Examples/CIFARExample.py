from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import ConnectionPatch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from FunctionEncoder import CIFARDataset, FunctionEncoder, ListCallback, TensorboardCallback, DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--representation_mode", type=str, default="least_squares",
                    choices=["least_squares", "inner_product", "encoder_network"],
                    help="Method for computing representation.")
parser.add_argument("--epochs", type=int, default=1_000)
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
    logdir = f"logs/cifar_example/{representation_mode}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# create a dataset
dataset = CIFARDataset()

if load_path is None:
    # Define default encoder kwargs (can be customized) - added for consistency
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean"
    )
    # create the model
    # Change 'method' to 'representation_mode' and add encoder_kwargs
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode, # Use correct argument name
                            model_type="CNN",
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(), # Add encoder kwargs
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters())) # Added print statement

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
    # Define default encoder kwargs for loading as well - added for consistency
    custom_encoder_kwargs = dict(
        phi_hidden_size=128,
        phi_n_layers=3,
        rho_hidden_size=128,
        rho_n_layers=3,
        activation="relu",
        aggregation="mean"
    )
    # Change 'method' to 'representation_mode' and add encoder_kwargs for loading
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            representation_mode=representation_mode, # Use correct argument name
                            model_type="CNN", # Need to specify model_type when loading too if it's not default
                            encoder_kwargs=custom_encoder_kwargs if representation_mode == "encoder_network" else dict(), # Add encoder kwargs
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot

# create a plot with 4 rows, 9 columns
# each row is a different class
# the first 4 columns are positive examples.
# the second 4 columns are negative examples.
# the last column is a new image from the class and its prediction
def plot(example_xs, y_hats, info, dataset, logdir, filename,):
    fig, ax = plt.subplots(4, 12, figsize=(18, 8), gridspec_kw={'width_ratios': [1,1,1,1,0.2,1,1,1,1,0.2,1, 1]})
    for row in range(4):
        # positive examples
        for col in range(4):
            ax[row, col].axis("off")
            img = example_xs[row, col].permute(2,1,0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["positive_example_class_indicies"][row]
            class_name = dataset.classes[class_idx]
            ax[row, col].set_title(class_name)

        # negative examples
        for col in range(5, 9):
            ax[row, col].axis("off")
            img = example_xs[row, -col + 4].permute(2,1,0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["negative_example_class_indicies"][row, -col+4]
            class_name = dataset.classes[class_idx]
            ax[row, col].set_title(class_name)

        # disable axis for the two unfilled plots
        ax[row, 4].axis("off")
        ax[row, 9].axis("off")

        # new image and prediction
        ax[row, 10].axis("off")
        img = xs[row, 0].permute(2,1,0).cpu().numpy()
        ax[row, 10].imshow(img)

        logits = y_hats[row, 0]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 10].set_title(f"$P(x \in C) = {probs[0].item()*100:.0f}\%$")

        # add new negative image and prediction
        ax[row, 11].axis("off")
        img = xs[row, -1].permute(2,1,0).cpu().numpy()
        ax[row, 11].imshow(img)

        logits = y_hats[row, -1]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 11].set_title(f"$P(x \in C) = {probs[0].item()*100:.0f}\%$")

    # add dashed lines between positive and negative examples
    left = ax[0, 3].get_position().xmax
    right = ax[0, 5].get_position().xmin
    xpos = (left+right) / 2
    top = ax[0, 3].get_position().ymax + 0.05
    bottom = ax[3, 3].get_position().ymin
    line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--")

    # add dashed lines between negative examples and new image
    left = ax[0, 8].get_position().xmax
    right = ax[0, 10].get_position().xmin
    xpos = (left+right) / 2
    line2 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--")

    fig.lines = line1, line2,

    # add one text above positive samples
    left = ax[0, 0].get_position().xmin
    right = ax[0, 3].get_position().xmax
    xpos = (left+right) / 2
    ypos = ax[0, 0].get_position().ymax + 0.08
    fig.text(xpos, ypos, "Positive Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above negative samples
    left = ax[0, 5].get_position().xmin
    right = ax[0, 8].get_position().xmax
    xpos = (left+right) / 2
    fig.text(xpos, ypos, "Negative Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above new image
    left = ax[0, 10].get_position().xmin
    right = ax[0, 11].get_position().xmax
    xpos = (left+right) / 2
    fig.text(xpos, ypos, "New Image", ha="center", va="center", fontsize=16, weight="bold")


    plt.savefig(f"{logdir}/{filename}.png")
    plt.clf()

# Function to calculate and return average metrics
def calculate_metrics(y_true, y_pred_logits):
    """Calculates accuracy and AUC for each concept in the batch and returns the average."""
    accuracies = []
    aucs = []

    # Ensure tensors are on CPU and detached from graph
    y_true_np = y_true.detach().cpu().numpy() # Shape: (n_concepts, n_query, 2), Values: [-5, 5]
    y_pred_logits_np = y_pred_logits.detach().cpu().numpy() # Shape: (n_concepts, n_query, 2)

    # Check the shape of logits - expecting (..., 2) for binary classification logits
    if y_pred_logits_np.shape[-1] != 2:
        print(f"Warning: Expected logits shape to end in 2, but got {y_pred_logits_np.shape}. Metrics might be incorrect.")
        # Cannot proceed if shape is wrong
        return float('nan'), float('nan')

    # Predicted labels based on argmax of the two logits
    y_pred_labels = np.argmax(y_pred_logits_np, axis=-1) # Shape: (n_concepts, n_query)

    # Calculate probabilities using softmax for AUC
    probs = torch.softmax(torch.from_numpy(y_pred_logits_np), dim=-1).numpy()
    # Probability of the positive class (assuming index 1 is positive)
    y_pred_prob_positive = probs[..., 1] # Shape: (n_concepts, n_query)

    # Convert true labels from [-5, 5] format to binary {0, 1} using argmax
    # Assumes the index with value 5 is the true class
    y_true_binary = np.argmax(y_true_np, axis=-1) # Shape: (n_concepts, n_query)

    num_concepts = y_true_np.shape[0]
    for i in range(num_concepts):
        # Get the data for the current concept
        current_y_true = y_true_binary[i] # Shape: (n_query,) - Already binary
        current_y_pred_labels = y_pred_labels[i] # Shape: (n_query,)
        current_y_pred_prob_positive = y_pred_prob_positive[i] # Shape: (n_query,)

        # Ensure inputs are 1D arrays for sklearn metrics, even if n_query=1
        if current_y_true.ndim == 0:
             current_y_true = np.array([current_y_true.item()])
             current_y_pred_labels = np.array([current_y_pred_labels.item()])
             current_y_pred_prob_positive = np.array([current_y_pred_prob_positive.item()])

        # Calculate accuracy
        try:
            # Inputs should now be compatible binary arrays
            acc = accuracy_score(current_y_true, current_y_pred_labels)
            accuracies.append(acc)
        except ValueError as e:
             # This error should ideally not happen now, but keep for safety
             print(f"Error calculating accuracy for concept {i}: {e}")
             print(f"  y_true shape: {current_y_true.shape}, dtype: {current_y_true.dtype}, unique: {np.unique(current_y_true)}")
             print(f"  y_pred shape: {current_y_pred_labels.shape}, dtype: {current_y_pred_labels.dtype}, unique: {np.unique(current_y_pred_labels)}")
             continue # Skip to next concept


        # Calculate AUC - requires at least one sample from each class
        try:
            if len(np.unique(current_y_true)) > 1:
                 # Ensure true labels are binary {0, 1}
                 auc = roc_auc_score(current_y_true, current_y_pred_prob_positive)
                 aucs.append(auc)
            else:
                 # AUC is not defined if only one class is present in y_true
                 pass
        except ValueError as e:
            print(f"Warning: Could not calculate AUC for concept {i}. Error: {e}")
            # Handle cases where probabilities might cause issues


    avg_accuracy = np.mean(accuracies) if accuracies else float('nan')
    avg_auc = np.mean(aucs) if aucs else float('nan') # Average only defined AUCs

    return avg_accuracy, avg_auc

# get a new dataset for testing
dataset = CIFARDataset(split="test")

# ID test
print("--- In-Distribution Test ---")
example_xs, example_ys, xs, ys, info = dataset.sample()
# Ensure predict_from_examples doesn't have 'method' argument
y_hats = model.predict_from_examples(example_xs, example_ys, xs)
# Calculate and print metrics
id_accuracy, id_auc = calculate_metrics(ys, y_hats)
print(f"Average ID Accuracy: {id_accuracy:.4f}")
print(f"Average ID AUC: {id_auc:.4f}")
# Plot results
plot(example_xs, y_hats, info, dataset, logdir, "in_distribution")


# OOD Test
print("\n--- Out-of-Distribution Test ---")
example_xs, example_ys, xs, ys, info = dataset.sample(heldout=True) # heldout classes
# Ensure predict_from_examples doesn't have 'method' argument
y_hats = model.predict_from_examples(example_xs, example_ys, xs)
# Calculate and print metrics
ood_accuracy, ood_auc = calculate_metrics(ys, y_hats)
print(f"Average OOD Accuracy: {ood_accuracy:.4f}")
print(f"Average OOD AUC: {ood_auc:.4f}")
# Plot results
plot(example_xs, y_hats, info, dataset, logdir, "out_of_distribution")