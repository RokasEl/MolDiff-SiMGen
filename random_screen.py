import yaml
import numpy as np
import os
import sys

# Parameters to randomly screen
num_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
mode = sys.argv[2] if len(sys.argv) > 2 else "w"  # Default to overwrite mode
output_dir = "screen_configs"
os.makedirs(output_dir, exist_ok=True)

# Initialize RNG for reproducibility
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
rng = np.random.default_rng(seed)

low = 1e-2
high = 1e-1

abs_low = 1e-2
abs_high = 0.125

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
    # Sample guidance_strength in log scale
    scale_mode = rng.choice(["fractional", "absolute"])
    if scale_mode == "fractional":
        guidance_strength = low * (high / low) ** rng.uniform(0, 1)
    elif scale_mode == "absolute":
        guidance_strength = abs_low * (abs_high / abs_low) ** rng.uniform(0, 1)

    # Sample bond_guidance_strength
    bond_guidance_strength = 0 if rng.random() < 0.5 else 1e-4
    default_sigma = float(rng.uniform(0.05, 0.8))
    config = {
        "guidance_strength": float(guidance_strength),
        "scale_mode": str(scale_mode),
        "default_sigma": default_sigma,
        "element_sigmas": {
            7: rng.uniform(0.01,1)*default_sigma,
            16: rng.uniform(0.01,1)*default_sigma,
        },
        "bond_guidance_strength": bond_guidance_strength,
        "num_mols": 1280,
        "batch_size": 128,
        "experiment_name": f"exp_{i}",
        "num_replicas": 1,
    }

    with open(os.path.join(output_dir, f"config_{i}.yaml"), "w") as f:
        yaml.safe_dump(config, f)

print(
    f"Configs successfully {'appended' if mode == 'a' else 'written'} to {output_dir}"
)
