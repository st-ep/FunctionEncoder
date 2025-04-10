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
REGULARIZATION_PARAM = 1e-3

# Benchmark Parameters
# Benchmark 1: Varying n_basis
N_BASIS_VALUES = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] # Test different numbers of basis functions
FIXED_N_FUNCTIONS_B1 = 100 # Fixed batch size for n_basis benchmark

# Benchmark 2: Varying N_FUNCTIONS (Batch Size)
N_FUNCTIONS_VALUES = [10, 25, 50, 100, 200, 400, 600, 800, 1000] # Test different batch sizes
FIXED_N_BASIS_B2 = 50 # Fixed n_basis for batch size benchmark

N_REPEATS = 5 # Reduced repeats slightly as runs are longer
N_WARMUP = 2   # Reduced warmups slightly

# Method Configurations
METHODS = {
    "least_squares": {
        "name": "Least Squares",
        "init_kwargs": {
            "representation_mode": "least_squares",
            "model_type": MLP,
            "model_kwargs": MODEL_KWARGS_MLP,
            "regularization_parameter": REGULARIZATION_PARAM,
            "use_residuals_method": False,
        },
        "marker": 'o',
    },
    "inner_product": {
        "name": "Inner Product",
        "init_kwargs": {
            "representation_mode": "inner_product",
            "model_type": MLP,
            "model_kwargs": MODEL_KWARGS_MLP,
            "regularization_parameter": REGULARIZATION_PARAM,
            "use_residuals_method": False,
        },
        "marker": '^',
    },
    "encoder_network": {
        "name": "Encoder Network (DeepSets)",
        "init_kwargs": {
            "representation_mode": "encoder_network",
            "model_type": MLP,
            "model_kwargs": MODEL_KWARGS_MLP,
            "encoder_type": RepresentationEncoderDeepSets,
            "encoder_kwargs": ENCODER_KWARGS_DEEPSETS,
            "use_residuals_method": False,
        },
        "marker": 's',
    }
}

# --- Data Generation Helper ---
def generate_dummy_data(n_functions, n_points, input_size, output_size, device):
    """Generates random data for timing purposes."""
    example_xs = torch.randn(n_functions, n_points, *input_size, device=device)
    example_ys = torch.randn(n_functions, n_points, *output_size, device=device)
    return example_xs, example_ys

# --- Helper Functions ---

def bytes_to_mib(byte_val):
    return byte_val / (1024 * 1024) if byte_val is not None and byte_val != float('inf') else byte_val

def initialize_model(method_key, n_basis, input_size, output_size, data_type, device):
    """Initializes a model based on the method key."""
    config = METHODS[method_key]
    model = FunctionEncoder(
        input_size=input_size,
        output_size=output_size,
        data_type=data_type,
        n_basis=n_basis,
        **config["init_kwargs"]
    ).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {config['name']} Param Count: {params:,}")
    return model

# --- Core Benchmarking Function ---

def benchmark_single_method(method_key, variable_param_name, variable_param_values,
                            fixed_n_basis, fixed_n_functions, n_example_points,
                            input_size, output_size, data_type, device):
    """Runs benchmark for a single method across variable parameters."""
    config = METHODS[method_key]
    print(f"\n--- Benchmarking Method: {config['name']} ---")
    results = {"time": [], "mem": []}

    for value in variable_param_values:
        current_n_basis = value if variable_param_name == "n_basis" else fixed_n_basis
        current_n_functions = value if variable_param_name == "N_FUNCTIONS" else fixed_n_functions

        print(f"  Testing {variable_param_name} = {value} (n_basis={current_n_basis}, N_FUNCTIONS={current_n_functions})")
        torch.cuda.empty_cache()
        model = None
        example_xs, example_ys = None, None
        avg_time = float('inf')
        peak_mem = float('inf')
        oom_flag = False

        try:
            # 1. Initialize Model for this iteration
            model = initialize_model(method_key, current_n_basis, input_size, output_size, data_type, device)

            # 2. Generate Data
            example_xs, example_ys = generate_dummy_data(
                current_n_functions, n_example_points, input_size, output_size, device
            )

            # 3. Warmup (isolated)
            print("    Warming up...")
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    _ = model.compute_representation(example_xs, example_ys)
                    if device == torch.device("cuda"): torch.cuda.synchronize()

            # 4. Measurement (isolated)
            print(f"    Running {N_REPEATS} repeats...")
            current_times = []
            current_mem_peaks = []
            with torch.no_grad():
                # Reset peak memory stats before the first measured run
                if device == torch.device("cuda"): torch.cuda.reset_peak_memory_stats(device)

                for i in range(N_REPEATS):
                    start_time = time.perf_counter()
                    _ = model.compute_representation(example_xs, example_ys)
                    if device == torch.device("cuda"): torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    current_times.append(end_time - start_time)

                    # Capture peak memory after the first repeat
                    if i == 0:
                        if device == torch.device("cuda"):
                            current_mem_peaks.append(torch.cuda.max_memory_allocated(device))
                        else:
                            current_mem_peaks.append(0) # Placeholder for CPU

            avg_time = np.mean(current_times)
            peak_mem = np.mean(current_mem_peaks) # Should only be one value if captured on i==0

        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM during {variable_param_name}={value} execution: {e}")
            oom_flag = True
            # avg_time and peak_mem remain 'inf'
        except Exception as e:
            print(f"    Error during {variable_param_name}={value} execution: {e}")
            oom_flag = True
            # avg_time and peak_mem remain 'inf'
        finally:
            # Cleanup for this iteration
            del model
            del example_xs
            del example_ys
            torch.cuda.empty_cache()

        results["time"].append(avg_time)
        results["mem"].append(peak_mem)

        # Print results for this step
        time_str = f"{avg_time:.6f}" if not oom_flag else "OOM"
        mem_mib = bytes_to_mib(peak_mem)
        mem_str = f"{mem_mib:.2f}" if not oom_flag else "OOM"
        print(f"    Avg Time: {time_str} seconds")
        print(f"    Peak Mem: {mem_str} MiB")

    return results

# --- Run Benchmarks ---

all_results_n_basis = {}
all_results_n_functions = {}

# Benchmark 1: Varying n_basis
print("="*20)
print("Starting Benchmark 1: Varying n_basis")
print(f"(Fixed N_FUNCTIONS = {FIXED_N_FUNCTIONS_B1}, N_EXAMPLE_POINTS = {N_EXAMPLE_POINTS})")
print("="*20)
for method_key in METHODS:
    all_results_n_basis[method_key] = benchmark_single_method(
        method_key=method_key,
        variable_param_name="n_basis",
        variable_param_values=N_BASIS_VALUES,
        fixed_n_basis=None, # Not used when varying n_basis
        fixed_n_functions=FIXED_N_FUNCTIONS_B1,
        n_example_points=N_EXAMPLE_POINTS,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        data_type=DATA_TYPE,
        device=DEVICE
    )
print("\nBenchmark 1 (n_basis) finished.")


# Benchmark 2: Varying N_FUNCTIONS
print("\n" + "="*20)
print("Starting Benchmark 2: Varying N_FUNCTIONS (Batch Size)")
print(f"(Fixed n_basis = {FIXED_N_BASIS_B2}, N_EXAMPLE_POINTS = {N_EXAMPLE_POINTS})")
print("="*20)
for method_key in METHODS:
    all_results_n_functions[method_key] = benchmark_single_method(
        method_key=method_key,
        variable_param_name="N_FUNCTIONS",
        variable_param_values=N_FUNCTIONS_VALUES,
        fixed_n_basis=FIXED_N_BASIS_B2,
        fixed_n_functions=None, # Not used when varying N_FUNCTIONS
        n_example_points=N_EXAMPLE_POINTS,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        data_type=DATA_TYPE,
        device=DEVICE
    )
print("\nBenchmark 2 (N_FUNCTIONS) finished.")


# --- Plotting ---
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Time vs n_basis ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # Subplot 1 for time

for method_key, results in all_results_n_basis.items():
    config = METHODS[method_key]
    times = results["time"]
    plot_x = [n for n, t in zip(N_BASIS_VALUES, times) if t != float('inf')]
    plot_y = [t for t in times if t != float('inf')]
    label = config['name']
    if len(plot_x) < len(N_BASIS_VALUES):
        oom_val = N_BASIS_VALUES[len(plot_x)]
        label += f' (OOM at n_basis={oom_val})'
    plt.plot(plot_x, plot_y, marker=config['marker'], linestyle='-', label=label)

plt.xlabel("Number of Basis Functions (n_basis)")
plt.ylabel(f"Avg Inference Time / Batch (s)\n(Batch={FIXED_N_FUNCTIONS_B1}, Pts/Func={N_EXAMPLE_POINTS})")
plt.title(f"Computation Speed vs n_basis ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_BASIS_VALUES)

# --- Plotting Memory vs n_basis ---
plt.subplot(1, 2, 2) # Subplot 2 for memory

for method_key, results in all_results_n_basis.items():
    config = METHODS[method_key]
    mems = results["mem"]
    plot_x = [n for n, m in zip(N_BASIS_VALUES, mems) if m != float('inf')]
    plot_y = [bytes_to_mib(m) for m in mems if m != float('inf')]
    label = config['name']
    if len(plot_x) < len(N_BASIS_VALUES):
        oom_val = N_BASIS_VALUES[len(plot_x)]
        label += f' (OOM at n_basis={oom_val})'
    plt.plot(plot_x, plot_y, marker=config['marker'], linestyle='-', label=label)


plt.xlabel("Number of Basis Functions (n_basis)")
plt.ylabel(f"Peak GPU Memory / Batch (MiB)\n(Batch={FIXED_N_FUNCTIONS_B1}, Pts/Func={N_EXAMPLE_POINTS})")
plt.title(f"Peak Memory vs n_basis ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_BASIS_VALUES)

plt.tight_layout()
plot_filename_n_basis = os.path.join(output_dir, f"inference_speed_memory_vs_nbasis_{DEVICE.type}.png")
plt.savefig(plot_filename_n_basis)
print(f"\nPlot saved to {plot_filename_n_basis}")


# --- Plotting Time vs N_FUNCTIONS ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # Subplot 1 for time

for method_key, results in all_results_n_functions.items():
    config = METHODS[method_key]
    times = results["time"]
    plot_x = [n for n, t in zip(N_FUNCTIONS_VALUES, times) if t != float('inf')]
    plot_y = [t for t in times if t != float('inf')]
    label = config['name']
    if len(plot_x) < len(N_FUNCTIONS_VALUES):
        oom_val = N_FUNCTIONS_VALUES[len(plot_x)]
        label += f' (OOM at N_FUNCTIONS={oom_val})'
    plt.plot(plot_x, plot_y, marker=config['marker'], linestyle='-', label=label)

plt.xlabel("Batch Size (N_FUNCTIONS)")
plt.ylabel(f"Avg Inference Time / Batch (s)\n(n_basis={FIXED_N_BASIS_B2}, Pts/Func={N_EXAMPLE_POINTS})")
plt.title(f"Computation Speed vs Batch Size ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_FUNCTIONS_VALUES)

# --- Plotting Memory vs N_FUNCTIONS ---
plt.subplot(1, 2, 2) # Subplot 2 for memory

for method_key, results in all_results_n_functions.items():
    config = METHODS[method_key]
    mems = results["mem"]
    plot_x = [n for n, m in zip(N_FUNCTIONS_VALUES, mems) if m != float('inf')]
    plot_y = [bytes_to_mib(m) for m in mems if m != float('inf')]
    label = config['name']
    if len(plot_x) < len(N_FUNCTIONS_VALUES):
        oom_val = N_FUNCTIONS_VALUES[len(plot_x)]
        label += f' (OOM at N_FUNCTIONS={oom_val})'
    plt.plot(plot_x, plot_y, marker=config['marker'], linestyle='-', label=label)

plt.xlabel("Batch Size (N_FUNCTIONS)")
plt.ylabel(f"Peak GPU Memory / Batch (MiB)\n(n_basis={FIXED_N_BASIS_B2}, Pts/Func={N_EXAMPLE_POINTS})")
plt.title(f"Peak Memory vs Batch Size ({DEVICE.type.upper()})")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.xticks(N_FUNCTIONS_VALUES)

plt.tight_layout()
plot_filename_n_funcs = os.path.join(output_dir, f"inference_speed_memory_vs_nfuncs_{DEVICE.type}.png")
plt.savefig(plot_filename_n_funcs)
print(f"Plot saved to {plot_filename_n_funcs}")

plt.show()
