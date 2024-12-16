import yaml
import numpy as np
import os
import sys

# Parameters to randomly screen
num_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
output_dir = "screen_configs"
os.makedirs(output_dir, exist_ok=True)

for i in range(num_configs):
    config = {
        "guidance_strength": float(np.random.uniform(0.0, 1.0)),
        "default_sigma": float(np.random.uniform(0.05, 0.8)),
        "element_sigmas": {7: float(np.random.uniform(0.1, 0.5))},
        "num_mols": 1280,
        "batch_size": 128,
        "experiment_name": f"exp_{i}"
    }
    with open(os.path.join(output_dir, f"config_{i}.yaml"), "w") as f:
        yaml.safe_dump(config, f)
