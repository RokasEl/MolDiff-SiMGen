import os
import sys

import numpy as np
import yaml

# Parameters to randomly screen
num_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
mode = sys.argv[2] if len(sys.argv) > 2 else "w"  # Default to overwrite mode
output_dir = "screen_configs"
os.makedirs(output_dir, exist_ok=True)

# Initialize RNG for reproducibility
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
rng = np.random.default_rng(seed)

guidance_mode = "direct"

if guidance_mode == "inverse":
    # Inverse guidance screen parameters
    low = 0.2
    high = 0.5
    min_low = 0.1
    min_high = 0.2

    abs_low = 0.05
    abs_high = 0.2
elif guidance_mode == "direct":
    low = 0.05
    high = 0.2
    min_low = 0.0
    min_high = 0.05

    abs_low = 0.01
    abs_high = 0.12


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
    # scale_mode = rng.choice(["fractional", "absolute"])
    scale_mode = "fractional"
    guidance_strength = (
        rng.uniform(low, high)
        if scale_mode == "fractional"
        else rng.uniform(abs_low, abs_high)
    )
    min_gui_strength = rng.uniform(min_low, min_high)

    default_sigma = 1.0
    # sigma_schedule = rng.choice(
    #     ["matching", "constant"]
    # )  # Refers to how sigma changes over time
    sigma_schedule = "matching"
    constant_sigma_value = rng.uniform(0.5, 2)

    is_frequency = [25, 50, 100][i%3]
    inverse_temperature = rng.uniform(1e-3, 1e-2)
    mini_batch = rng.choice([4, 8, 16])

    config = {
        "experiment_name": f"exp_{i}",
        "results_dir": "./results/" if guidance_mode == "direct" else "./results_inverse_summation/",
        "batch_size": 128,
        "num_mols": 512,
        "max_size": 20,
        "guidance_mode": guidance_mode,  
        "guidance_strength": float(guidance_strength),
        "min_gui_scale": float(min_gui_strength),
        "scale_mode": str(scale_mode),
        "bond_guidance_strength": 0.0,  # Known to not help from previous runs
        "default_sigma": default_sigma,  # Element sigma for all elements
        "element_sigmas": {
            7: rng.uniform(1.0, 1.5) * default_sigma,
            16: rng.uniform(1.0, 1.5) * default_sigma,
        },
        "sigma_schedule": str(sigma_schedule),
        "constant_sigma_value": float(constant_sigma_value),
        "importance_sampling_freq": int(is_frequency),
        "inverse_temperature": float(inverse_temperature),
        "mini_batch": int(mini_batch),
    }

    with open(os.path.join(output_dir, f"config_{i}.yaml"), "w") as f:
        yaml.safe_dump(config, f)

print(
    f"Configs successfully {'appended' if mode == 'a' else 'written'} to {output_dir}"
)