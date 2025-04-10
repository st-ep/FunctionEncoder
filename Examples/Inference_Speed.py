import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root to the Python path
# This assumes the script is run from the 'Examples' directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from FunctionEncoder.Model.Architecture.RepresentationEncoderDeepSets import RepresentationEncoderDeepSets
from FunctionEncoder.Model.Architecture.MLP import MLP
# Assuming you have a dataset generator, e.g., GaussianProcessDataset
# If not, we can create simple dummy data for timing purposes.
# from FunctionEncoder.Dataset.GaussianProcessDataset import GaussianProcessDataset

# --- Configuration ---
INPUT_SIZE = (1,)
OUTPUT_SIZE = (1,)
DATA_TYPE = "deterministic" # Affects inner product, shouldn't drastically change timing difference pattern
# N_FUNCTIONS = 100       # Batch size for inference - Now varied in the second benchmark
# N_EXAMPLE_POINTS = 10000 # Number of (x, y) pairs per function - REDUCED TO SHOW SCALING BEFORE OOM
N_EXAMPLE_POINTS = 2000 # Number of (x, y) pairs per function
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE == torch.device("cpu"):
    print("WARNING: Memory benchmarking is most relevant on CUDA devices.")

# Model Hyperparameters (adjust to try and match parameter counts if desired)
# Common basis model params
MODEL_KWARGS_MLP = {
    "hidden_size": 128,
    "n_layers": 3,
    "activation": "relu"
}
# Encoder specific params
ENCODER_KWARGS_DEEPSETS = {
    "phi_hidden_size": 128,
    "phi_n_layers": 3,
    "rho_hidden_size": 128,
    "rho_n_layers": 3,
    "activation": "relu",
    "aggregation": "mean", # Or 'attention', 'sum', 'max'
    "use_layer_norm": False
}

# Benchmark Parameters
# Benchmark 1: Varying n_basis
N_BASIS_VALUES = [10, 25, 50, 100, 150, 200] # Test different numbers of basis functions
FIXED_N_FUNCTIONS = 100 # Fixed batch size for n_basis benchmark

# Benchmark 2: Varying N_FUNCTIONS (Batch Size)
N_FUNCTIONS_VALUES = [10, 25, 50, 100, 200, 400] # Test different batch sizes
FIXED_N_BASIS = 50 # Fixed n_basis for batch size benchmark

N_REPEATS = 5 # Reduced repeats slightly as runs are longer
N_WARMUP = 2   # Reduced warmups slightly

# --- Data Generation Helper ---
def generate_dummy_data(n_functions, n_points, input_size, output_size, device):
    """Generates random data for timing purposes."""
    example_xs = torch.randn(n_functions, n_points, *input_size, device=device)
    example_ys = torch.randn(n_functions, n_points, *output_size, device=device)
    return example_xs, example_ys

# --- Benchmarking ---
# Results storage for n_basis benchmark
results_n_basis = {
    "least_squares_time": [],
    "encoder_network_time": [],
    "least_squares_mem": [],
    "encoder_network_mem": []
}
param_counts_n_basis = {
    "least_squares": [],
    "encoder_network": []
}

# Results storage for n_functions benchmark
results_n_functions = {
    "least_squares_time": [],
    "encoder_network_time": [],
    "least_squares_mem": [],
    "encoder_network_mem": []
}
# Parameter counts will be the same for all runs in the n_functions benchmark
param_counts_n_functions = {}


# Function to convert bytes to MiB
def bytes_to_mib(byte_val):
    return byte_val / (1024 * 1024) if byte_val is not None and byte_val != float('inf') else byte_val


# --- Benchmark 1: Varying n_basis ---
print("="*20)
print("Starting Benchmark 1: Varying n_basis")
print(f"(Fixed N_FUNCTIONS = {FIXED_N_FUNCTIONS}, N_EXAMPLE_POINTS = {N_EXAMPLE_POINTS})")
print("="*20)

for n_basis in N_BASIS_VALUES:
    print(f"--- Testing n_basis = {n_basis} ---")
    torch.cuda.empty_cache() # Clear cache before initializing models

    # --- Initialize Models ---
    # Least Squares Model
    model_ls = FunctionEncoder(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        data_type=DATA_TYPE,
        n_basis=n_basis,
        model_type=MLP, # Using MLP for basis functions
        model_kwargs=MODEL_KWARGS_MLP,
        representation_mode="least_squares",
        use_residuals_method=False,
        regularization_parameter=1e-3
    ).to(DEVICE)
    model_ls.eval()

    # Encoder Network Model
    model_enc = FunctionEncoder(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        data_type=DATA_TYPE,
        n_basis=n_basis,
        model_type=MLP, # Using MLP for basis functions
        model_kwargs=MODEL_KWARGS_MLP,
        representation_mode="encoder_network",
        encoder_type=RepresentationEncoderDeepSets,
        encoder_kwargs=ENCODER_KWARGS_DEEPSETS,
        use_residuals_method=False
    ).to(DEVICE)
    model_enc.eval()

    ls_params = sum(p.numel() for p in model_ls.parameters())
    enc_params = sum(p.numel() for p in model_enc.parameters())
    param_counts_n_basis["least_squares"].append(ls_params)
    param_counts_n_basis["encoder_network"].append(enc_params)
    print(f"  LS Param Count: {ls_params:,}")
    print(f"  Enc Param Count: {enc_params:,}")

    # --- Generate Data ---
    example_xs, example_ys = generate_dummy_data(
        FIXED_N_FUNCTIONS, N_EXAMPLE_POINTS, INPUT_SIZE, OUTPUT_SIZE, DEVICE # Use FIXED_N_FUNCTIONS
    )

    # --- Time and Memory Measurement ---
    ls_times = []
    enc_times = []
    ls_mem_peaks = []
    enc_mem_peaks = []
    ls_oom = False # Flag to track if LS hit OOM for this n_basis

    with torch.no_grad():
        # Warm-up runs (run both to load kernels, etc.)
        print("  Warming up...")
        try:
            for _ in range(N_WARMUP):
                _ = model_ls.compute_representation(example_xs, example_ys)
                _ = model_enc.compute_representation(example_xs, example_ys)
                if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
             print("  OOM during warmup, proceeding...") # LS might OOM here too
             ls_oom = True # Assume LS caused it if it happens here
        except Exception as e:
             print(f"  Error during warmup: {e}")
             # Decide how to handle other errors, maybe skip this n_basis

        # --- Least Squares Timing & Memory ---
        print(f"  Running LS {N_REPEATS} repeats...")
        if not ls_oom: # Only run if warmup didn't already fail
            try:
                # Reset memory stats specifically for LS run
                if DEVICE == torch.device("cuda"): torch.cuda.reset_peak_memory_stats(DEVICE)

                start_time = time.perf_counter()
                current_ls_mem_peaks = [] # Collect peaks for each repeat if needed, or just first
                for i in range(N_REPEATS):
                    _ = model_ls.compute_representation(example_xs, example_ys)
                    if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
                    # Record memory on first repeat after sync
                    if i == 0 and DEVICE == torch.device("cuda"):
                         current_ls_mem_peaks.append(torch.cuda.max_memory_allocated(DEVICE))
                    elif i == 0 and DEVICE == torch.device("cpu"):
                         current_ls_mem_peaks.append(0) # Placeholder for CPU

                end_time = time.perf_counter()
                ls_times.append((end_time - start_time) / N_REPEATS)
                ls_mem_peaks.append(np.mean(current_ls_mem_peaks)) # Store the peak from the first run

            except torch.cuda.OutOfMemoryError:
                print("  OOM during Least Squares computation!")
                ls_oom = True
                ls_times.append(float('inf')) # Indicate failure
                ls_mem_peaks.append(float('inf'))
            except Exception as e:
                 print(f"  Error during LS computation: {e}")
                 ls_oom = True # Treat other errors as failure for simplicity
                 ls_times.append(float('inf'))
                 ls_mem_peaks.append(float('inf'))
        else:
             # If warmup failed, record failure for timing too
             ls_times.append(float('inf'))
             ls_mem_peaks.append(float('inf'))

        # --- Encoder Network Timing & Memory ---
        print(f"  Running Encoder {N_REPEATS} repeats...")
        try:
            # Reset memory stats specifically for Encoder run
            if DEVICE == torch.device("cuda"): torch.cuda.reset_peak_memory_stats(DEVICE)

            start_time = time.perf_counter()
            current_enc_mem_peaks = []
            for i in range(N_REPEATS):
                _ = model_enc.compute_representation(example_xs, example_ys)
                if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
                 # Record memory on first repeat after sync
                if i == 0 and DEVICE == torch.device("cuda"):
                     current_enc_mem_peaks.append(torch.cuda.max_memory_allocated(DEVICE))
                elif i == 0 and DEVICE == torch.device("cpu"):
                     current_enc_mem_peaks.append(0) # Placeholder for CPU

            end_time = time.perf_counter()
            enc_times.append((end_time - start_time) / N_REPEATS)
            enc_mem_peaks.append(np.mean(current_enc_mem_peaks)) # Store the peak from the first run

        except torch.cuda.OutOfMemoryError:
             # This shouldn't happen based on previous results, but handle it
             print("  OOM during Encoder computation!")
             enc_times.append(float('inf'))
             enc_mem_peaks.append(float('inf'))
        except Exception as e:
             print(f"  Error during Encoder computation: {e}")
             enc_times.append(float('inf'))
             enc_mem_peaks.append(float('inf'))


    # --- Store Results ---
    avg_ls_time = np.mean(ls_times) # Will be inf if OOM occurred
    avg_enc_time = np.mean(enc_times)
    peak_ls_mem = np.mean(ls_mem_peaks) # Use mean, will be inf if OOM
    peak_enc_mem = np.mean(enc_mem_peaks)

    results_n_basis["least_squares_time"].append(avg_ls_time)
    results_n_basis["encoder_network_time"].append(avg_enc_time)
    results_n_basis["least_squares_mem"].append(peak_ls_mem)
    results_n_basis["encoder_network_mem"].append(peak_enc_mem)

    # --- Print Results for this n_basis ---
    ls_time_str = f"{avg_ls_time:.6f}" if avg_ls_time != float('inf') else "OOM"
    print(f"  Avg LS Time: {ls_time_str} seconds")
    enc_time_str = f"{avg_enc_time:.6f}" if avg_enc_time != float('inf') else "OOM"
    print(f"  Avg Enc Time: {enc_time_str} seconds")
    ls_mem_mib = bytes_to_mib(peak_ls_mem)
    ls_mem_str = f"{ls_mem_mib:.2f}" if ls_mem_mib != float('inf') else "OOM"
    print(f"  Peak LS Mem: {ls_mem_str} MiB")
    enc_mem_mib = bytes_to_mib(peak_enc_mem)
    enc_mem_str = f"{enc_mem_mib:.2f}" if enc_mem_mib != float('inf') else "OOM"
    print(f"  Peak Enc Mem: {enc_mem_str} MiB")

    # Clean up models to free memory before next iteration
    del model_ls
    del model_enc
    del example_xs
    del example_ys
    if DEVICE == torch.device("cuda"): torch.cuda.empty_cache()

print("Benchmark 1 (n_basis) finished.")


# --- Benchmark 2: Varying N_FUNCTIONS (Batch Size) ---
print("\n" + "="*20)
print("Starting Benchmark 2: Varying N_FUNCTIONS (Batch Size)")
print(f"(Fixed n_basis = {FIXED_N_BASIS}, N_EXAMPLE_POINTS = {N_EXAMPLE_POINTS})")
print("="*20)

# Initialize models ONCE outside the loop for this benchmark
# as parameter counts don't change with N_FUNCTIONS
model_ls_base = FunctionEncoder(
    input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, data_type=DATA_TYPE,
    n_basis=FIXED_N_BASIS, model_type=MLP, model_kwargs=MODEL_KWARGS_MLP,
    representation_mode="least_squares", use_residuals_method=False,
    regularization_parameter=1e-3
).to(DEVICE)
model_ls_base.eval()

model_enc_base = FunctionEncoder(
    input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, data_type=DATA_TYPE,
    n_basis=FIXED_N_BASIS, model_type=MLP, model_kwargs=MODEL_KWARGS_MLP,
    representation_mode="encoder_network", encoder_type=RepresentationEncoderDeepSets,
    encoder_kwargs=ENCODER_KWARGS_DEEPSETS, use_residuals_method=False
).to(DEVICE)
model_enc_base.eval()

param_counts_n_functions["least_squares"] = sum(p.numel() for p in model_ls_base.parameters())
param_counts_n_functions["encoder_network"] = sum(p.numel() for p in model_enc_base.parameters())
print(f"Models initialized with n_basis = {FIXED_N_BASIS}")
print(f"  LS Param Count: {param_counts_n_functions['least_squares']:,}")
print(f"  Enc Param Count: {param_counts_n_functions['encoder_network']:,}")


for n_funcs in N_FUNCTIONS_VALUES:
    print(f"--- Testing N_FUNCTIONS = {n_funcs} ---")
    torch.cuda.empty_cache() # Clear cache before generating data

    # --- Generate Data ---
    # Use the current n_funcs from the loop
    example_xs, example_ys = generate_dummy_data(
        n_funcs, N_EXAMPLE_POINTS, INPUT_SIZE, OUTPUT_SIZE, DEVICE
    )

    # --- Time and Memory Measurement ---
    ls_times = []
    enc_times = []
    ls_mem_peaks = []
    enc_mem_peaks = []
    ls_oom = False # Flag to track if LS hit OOM for this n_funcs

    with torch.no_grad():
        # Warm-up runs
        print("  Warming up...")
        try:
            for _ in range(N_WARMUP):
                _ = model_ls_base.compute_representation(example_xs, example_ys)
                _ = model_enc_base.compute_representation(example_xs, example_ys)
                if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
             print("  OOM during warmup, proceeding...")
             ls_oom = True # Assume LS caused it
        except Exception as e:
             print(f"  Error during warmup: {e}")
             # Handle other errors if necessary

        # --- Least Squares Timing & Memory ---
        print(f"  Running LS {N_REPEATS} repeats...")
        if not ls_oom:
            try:
                if DEVICE == torch.device("cuda"): torch.cuda.reset_peak_memory_stats(DEVICE)
                start_time = time.perf_counter()
                current_ls_mem_peaks = []
                for i in range(N_REPEATS):
                    _ = model_ls_base.compute_representation(example_xs, example_ys)
                    if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
                    if i == 0 and DEVICE == torch.device("cuda"):
                         current_ls_mem_peaks.append(torch.cuda.max_memory_allocated(DEVICE))
                    elif i == 0 and DEVICE == torch.device("cpu"):
                         current_ls_mem_peaks.append(0)
                end_time = time.perf_counter()
                ls_times.append((end_time - start_time) / N_REPEATS)
                ls_mem_peaks.append(np.mean(current_ls_mem_peaks))
            except torch.cuda.OutOfMemoryError:
                print("  OOM during Least Squares computation!")
                ls_oom = True
                ls_times.append(float('inf'))
                ls_mem_peaks.append(float('inf'))
            except Exception as e:
                 print(f"  Error during LS computation: {e}")
                 ls_oom = True
                 ls_times.append(float('inf'))
                 ls_mem_peaks.append(float('inf'))
        else:
             ls_times.append(float('inf'))
             ls_mem_peaks.append(float('inf'))

        # --- Encoder Network Timing & Memory ---
        print(f"  Running Encoder {N_REPEATS} repeats...")
        try:
            if DEVICE == torch.device("cuda"): torch.cuda.reset_peak_memory_stats(DEVICE)
            start_time = time.perf_counter()
            current_enc_mem_peaks = []
            for i in range(N_REPEATS):
                _ = model_enc_base.compute_representation(example_xs, example_ys)
                if DEVICE == torch.device("cuda"): torch.cuda.synchronize()
                if i == 0 and DEVICE == torch.device("cuda"):
                     current_enc_mem_peaks.append(torch.cuda.max_memory_allocated(DEVICE))
                elif i == 0 and DEVICE == torch.device("cpu"):
                     current_enc_mem_peaks.append(0)
            end_time = time.perf_counter()
            enc_times.append((end_time - start_time) / N_REPEATS)
            enc_mem_peaks.append(np.mean(current_enc_mem_peaks))
        except torch.cuda.OutOfMemoryError:
             print("  OOM during Encoder computation!")
             enc_times.append(float('inf'))
             enc_mem_peaks.append(float('inf'))
        except Exception as e:
             print(f"  Error during Encoder computation: {e}")
             enc_times.append(float('inf'))
             enc_mem_peaks.append(float('inf'))

    # --- Store Results ---
    avg_ls_time = np.mean(ls_times)
    avg_enc_time = np.mean(enc_times)
    peak_ls_mem = np.mean(ls_mem_peaks)
    peak_enc_mem = np.mean(enc_mem_peaks)

    results_n_functions["least_squares_time"].append(avg_ls_time)
    results_n_functions["encoder_network_time"].append(avg_enc_time)
    results_n_functions["least_squares_mem"].append(peak_ls_mem)
    results_n_functions["encoder_network_mem"].append(peak_enc_mem)

    # --- Print Results for this n_funcs ---
    ls_time_str = f"{avg_ls_time:.6f}" if avg_ls_time != float('inf') else "OOM"
    print(f"  Avg LS Time: {ls_time_str} seconds")
    enc_time_str = f"{avg_enc_time:.6f}" if avg_enc_time != float('inf') else "OOM"
    print(f"  Avg Enc Time: {enc_time_str} seconds")
    ls_mem_mib = bytes_to_mib(peak_ls_mem)
    ls_mem_str = f"{ls_mem_mib:.2f}" if ls_mem_mib != float('inf') else "OOM"
    print(f"  Peak LS Mem: {ls_mem_str} MiB")
    enc_mem_mib = bytes_to_mib(peak_enc_mem)
    enc_mem_str = f"{enc_mem_mib:.2f}" if enc_mem_mib != float('inf') else "OOM"
    print(f"  Peak Enc Mem: {enc_mem_str} MiB")

    # Clean up data only, models persist
    del example_xs
    del example_ys
    if DEVICE == torch.device("cuda"): torch.cuda.empty_cache()

# Clean up base models after the loop
del model_ls_base
del model_enc_base
if DEVICE == torch.device("cuda"): torch.cuda.empty_cache()

print("Benchmark 2 (N_FUNCTIONS) finished.")


# --- Plotting ---
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Time vs n_basis ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # Create subplot 1 for time

# Filter out inf values for plotting LS time
plot_n_basis_ls_time = [n for n, t in zip(N_BASIS_VALUES, results_n_basis["least_squares_time"]) if t != float('inf')]
plot_ls_time = [t for t in results_n_basis["least_squares_time"] if t != float('inf')]
if len(plot_n_basis_ls_time) < len(N_BASIS_VALUES):
    ls_label = f'Least Squares (OOM at n_basis={N_BASIS_VALUES[len(plot_n_basis_ls_time)]})'
else:
    ls_label = 'Least Squares'

plt.plot(plot_n_basis_ls_time, plot_ls_time, marker='o', linestyle='-', label=ls_label)
plt.plot(N_BASIS_VALUES, results_n_basis["encoder_network_time"], marker='s', linestyle='-', label='Encoder Network (DeepSets)')

plt.xlabel("Number of Basis Functions (n_basis)")
plt.ylabel(f"Avg Inference Time / Batch (s)\n(Batch={FIXED_N_FUNCTIONS}, Pts/Func={N_EXAMPLE_POINTS})") # Use FIXED_N_FUNCTIONS
plt.title(f"Computation Speed vs n_basis ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_BASIS_VALUES)

# --- Plotting Memory vs n_basis ---
plt.subplot(1, 2, 2) # Create subplot 2 for memory

# Filter out inf values for plotting LS memory
plot_n_basis_ls_mem = [n for n, m in zip(N_BASIS_VALUES, results_n_basis["least_squares_mem"]) if m != float('inf')]
plot_ls_mem = [bytes_to_mib(m) for m in results_n_basis["least_squares_mem"] if m != float('inf')]
if len(plot_n_basis_ls_mem) < len(N_BASIS_VALUES):
     ls_mem_label = f'Least Squares (OOM at n_basis={N_BASIS_VALUES[len(plot_n_basis_ls_mem)]})'
else:
     ls_mem_label = 'Least Squares'

plt.plot(plot_n_basis_ls_mem, plot_ls_mem, marker='o', linestyle='-', label=ls_mem_label)
plt.plot(N_BASIS_VALUES, [bytes_to_mib(m) for m in results_n_basis["encoder_network_mem"]], marker='s', linestyle='-', label='Encoder Network (DeepSets)')

plt.xlabel("Number of Basis Functions (n_basis)")
plt.ylabel(f"Peak GPU Memory / Batch (MiB)\n(Batch={FIXED_N_FUNCTIONS}, Pts/Func={N_EXAMPLE_POINTS})") # Use FIXED_N_FUNCTIONS
plt.title(f"Peak Memory vs n_basis ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log') # Log scale often useful for memory too
plt.xticks(N_BASIS_VALUES)


plt.tight_layout() # Adjust layout to prevent overlap

# Save the n_basis plot
plot_filename_n_basis = os.path.join(output_dir, f"inference_speed_memory_vs_nbasis_{DEVICE.type}.png")
plt.savefig(plot_filename_n_basis)
print(f"Plot saved to {plot_filename_n_basis}")
# plt.show() # Show plots separately or together at the end


# --- Plotting Time vs N_FUNCTIONS ---
plt.figure(figsize=(12, 6)) # Create a new figure
plt.subplot(1, 2, 1) # Subplot 1 for time

# Filter out inf values for plotting LS time
plot_n_funcs_ls_time = [n for n, t in zip(N_FUNCTIONS_VALUES, results_n_functions["least_squares_time"]) if t != float('inf')]
plot_ls_time_funcs = [t for t in results_n_functions["least_squares_time"] if t != float('inf')]
if len(plot_n_funcs_ls_time) < len(N_FUNCTIONS_VALUES):
    ls_label_funcs = f'Least Squares (OOM at N_FUNCTIONS={N_FUNCTIONS_VALUES[len(plot_n_funcs_ls_time)]})'
else:
    ls_label_funcs = 'Least Squares'

plt.plot(plot_n_funcs_ls_time, plot_ls_time_funcs, marker='o', linestyle='-', label=ls_label_funcs)
plt.plot(N_FUNCTIONS_VALUES, results_n_functions["encoder_network_time"], marker='s', linestyle='-', label='Encoder Network (DeepSets)')

plt.xlabel("Batch Size (N_FUNCTIONS)")
plt.ylabel(f"Avg Inference Time / Batch (s)\n(n_basis={FIXED_N_BASIS}, Pts/Func={N_EXAMPLE_POINTS})") # Use FIXED_N_BASIS
plt.title(f"Computation Speed vs Batch Size ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_FUNCTIONS_VALUES) # Use batch size values for ticks

# --- Plotting Memory vs N_FUNCTIONS ---
plt.subplot(1, 2, 2) # Subplot 2 for memory

# Filter out inf values for plotting LS memory
plot_n_funcs_ls_mem = [n for n, m in zip(N_FUNCTIONS_VALUES, results_n_functions["least_squares_mem"]) if m != float('inf')]
plot_ls_mem_funcs = [bytes_to_mib(m) for m in results_n_functions["least_squares_mem"] if m != float('inf')]
if len(plot_n_funcs_ls_mem) < len(N_FUNCTIONS_VALUES):
     ls_mem_label_funcs = f'Least Squares (OOM at N_FUNCTIONS={N_FUNCTIONS_VALUES[len(plot_n_funcs_ls_mem)]})'
else:
     ls_mem_label_funcs = 'Least Squares'

plt.plot(plot_n_funcs_ls_mem, plot_ls_mem_funcs, marker='o', linestyle='-', label=ls_mem_label_funcs)
plt.plot(N_FUNCTIONS_VALUES, [bytes_to_mib(m) for m in results_n_functions["encoder_network_mem"]], marker='s', linestyle='-', label='Encoder Network (DeepSets)')

plt.xlabel("Batch Size (N_FUNCTIONS)")
plt.ylabel(f"Peak GPU Memory / Batch (MiB)\n(n_basis={FIXED_N_BASIS}, Pts/Func={N_EXAMPLE_POINTS})") # Use FIXED_N_BASIS
plt.title(f"Peak Memory vs Batch Size ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_FUNCTIONS_VALUES) # Use batch size values for ticks


plt.tight_layout()

# Save the N_FUNCTIONS plot
plot_filename_n_funcs = os.path.join(output_dir, f"inference_speed_memory_vs_nfuncs_{DEVICE.type}.png")
plt.savefig(plot_filename_n_funcs)
print(f"Plot saved to {plot_filename_n_funcs}")

plt.show() # Show all plots at the end
