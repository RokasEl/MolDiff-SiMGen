import os
import sys

import numpy as np
import yaml

# Parameters to randomly screen
num_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
mode = sys.argv[2] if len(sys.argv) > 2 else "w"  # Default to overwrite mode
output_dir = "screen_configs_inverse_summation"
os.makedirs(output_dir, exist_ok=True)

# Initialize RNG for reproducibility
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
rng = np.random.default_rng(seed)

low = 0.2
high = 0.5
min_low = 0.0
min_high = 0.1

abs_low = 0.1
abs_high = 0.3

# Determine starting index based on mode
if mode == "a":
    # Append mode: find the max index in existing configs
    existing_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("config_") and f.endswith(".yaml")
    ]
    max_idx = max(
        [int(f.split("_")[1].split(".")[0]) for f in existing_files], default=-1
    )
    start_idx = max_idx + 1
else:
    # Overwrite mode: start from index 0
    start_idx = 0

for i in range(start_idx, start_idx + num_configs):
    scale_mode = rng.choice(["fractional", "absolute"])
    guidance_strength = (
        rng.uniform(low, high)
        if scale_mode == "fractional"
        else rng.uniform(abs_low, abs_high)
    )
    min_gui_strength = rng.uniform(min_low, min_high)

    default_sigma = 1.0
    sigma_schedule = rng.choice(
        ["matching", "constant"]
    )  # refers to how sigma changes over time
    constant_sigma_value = rng.uniform(0.5, 2)

    is_frequency = rng.choice([25, 50, 100])
    inverse_temperature = rng.uniform(1e-3, 1e-2)
    mini_batch = rng.choice([8, 16])

    config = {
        "experiment_name": f"exp_{i}",
        "batch_size": 128,
        "num_mols": 1280,
        "max_size": 20,
        "guidance_strength": float(guidance_strength),
        "min_gui_scale": float(min_gui_strength),
        "scale_mode": str(scale_mode),
        "bond_guidance_strength": 0.0,  # Know this doesn't help from previous runs
        "default_sigma": default_sigma,  # this is the element sigma for all elements
        "element_sigmas": {
            7: rng.uniform(1.0, 1.5) * default_sigma,
            16: rng.uniform(1.0, 1.5) * default_sigma,
        },
        "sigma_schedule": str(sigma_schedule),
        "constant_sigma_value": constant_sigma_value,
        "importance_sampling_freq": is_frequency,
        "inverse_temperature": inverse_temperature,
        "mini_batch": mini_batch,
    }

    with open(os.path.join(output_dir, f"config_{i}.yaml"), "w") as f:
        yaml.safe_dump(config, f)

print(
    f"Configs successfully {'appended' if mode == 'a' else 'written'} to {output_dir}"
)
