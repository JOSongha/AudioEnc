# Training Speech X on SuperPOD 2

This folder (`spd2/`) contains files you need for launching Speech X training jobs on SuperPOD 2. **Important**: Run all commands from the **project root directory**, which is **one level above** the `spd2` folder.

The recommended config file for SPD2 is located at:
```
configs/speechx-v6/speechx-v6-s2-spd2.yaml
```
Adjust as needed or generate an alternative config via Jinja or other processes.

---

## Folder Contents

1. **`Dockerfile`**  
   - Defines the Docker/Enroot environment for Speech X training (based on `reg.navercorp.com/steering/audiollm-speechx`).
   - Modify as needed, rebuild, and push/convert to `.sqsh` if your environment requires updates.

2. **`bind.sh`**  
   - Optional script for CPU, memory, and IB device binding in multi-GPU Slurm jobs.
   - Called from the Slurm script (`audiollm-speechx.sbatch`) with default parameters (`--cpu=node`, etc.).

3. **`audiollm-speechx.sbatch`**  
   - The main Slurm batch script to run `llamafactory-cli train`.
   - Logs `srun` outputs in the `spd2/` folder, sets environment variables, and handles optional proxy settings.

4. **`run.sh`**  
   - A simple script that calls `torchrun` on `src/llamafactory/launcher.py`, passing a config file.
   - **Usage**: `bash spd2/run.sh [CONFIG_FILE]`  
     - If no `CONFIG_FILE` is provided, it defaults to `configs/speechx-v6-s2-spd2.yaml`.
   - This can be useful for local testing or non-Slurm runs (e.g., in a container or single-node environment).

5. **`callbacks.py`** (Optional)  
   - Contains a custom `WandbCallback` to enable **WandB resuming** or **forking** from a previous run.  
   - This callback checks environment variables like `WANDB_RESUME_RUN_ID` and the user’s `resume_from_checkpoint` argument, allowing you to continue or branch off from a prior WandB run with minimal manual setup.

6. **`README.md`**  
   - This documentation file.

---

## Quick Start

1. **Ensure You’re in the Project Root**  
   From your project’s top-level directory (the parent of `spd2`), you can see both `spd2/` and `configs/` subdirectories. All commands below assume this location.

2. **Build Dockerfile**  
   - If you **already** have a `.sqsh` image in `/mnt/fr20tb/audiollm/images` that you trust, you can **skip** this Docker build step (and the Enroot import step below) – just confirm your `.sbatch` script references the correct `.sqsh` file.
   - Otherwise, from the project root, change directory into `spd2`:
     ```bash
     cd spd2
     ```
   - Build the Docker image with an appropriate tag (e.g., `latest` or a semantic version):
     ```bash
     docker build -t reg.navercorp.com/steering/audiollm-speechx:latest .
     ```
   - Push the Docker image to your registry (if you have permissions):
     ```bash
     docker push reg.navercorp.com/steering/audiollm-speechx:latest
     ```
   - Return to the project root:
     ```bash
     cd ..
     ```

3. **Apply Enroot to the Docker Image**  
   - If you **do not** already have a `.sqsh` file for your environment in `/mnt/fr20tb/audiollm/images`, follow these steps:
     1. **Log into** the submission server of SuperPOD 2 (the cluster login node).
     2. **Pull** the Docker image from your registry:
        ```bash
        docker pull reg.navercorp.com/steering/audiollm-speechx:latest
        ```
     3. **Convert** the Docker image to an Enroot `.sqsh` file:
        ```bash
        enroot import -o /mnt/fr20tb/audiollm/images/audiollm-speechx-<version>.sqsh \
          dockerd://reg.navercorp.com/steering/audiollm-speechx:latest
        ```
   - This `.sqsh` image is used by Slurm’s container plugin in `audiollm-speechx.sbatch`.

4. **Pick or Adjust Your Config**  
   - The default SPD2 config is at: `configs/speechx-v6/speechx-v6-s2-spd2.yaml`.
   - If you need custom paths, environment variables, or hyperparameters, edit `configs/speechx-v6/speechx-v6-s2-spd2.yaml` accordingly.

5. **Set (Optional) Proxy Environment Variables**  
   - If you need a proxy server for external connections, use:
     ```bash
     export HTTP_PROXY="http://125.209.240.12:10038"
     export HTTPS_PROXY="http://125.209.240.12:10038"
     ```
   - If no proxy is required, skip this step.

Below is an **updated** README snippet with a **new note** in the **“Submit the Slurm Job”** section explaining how to resume from a checkpoint (independent of WandB). We also leave a brief mention in the WandB section about how forking interacts with resume steps.

```markdown
6. **Submit the Slurm Job**  
   - Provide your Weights & Biases API key, and optionally the proxy variables, then call `sbatch` against the script in `spd2/`:
     ```bash
     WANDB_API_KEY=<YOUR_KEY> \
     HTTP_PROXY="http://125.209.240.12:10038" \
     HTTPS_PROXY="http://125.209.240.12:10038" \
     sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
     ```
   - If no proxy is needed:
     ```bash
     WANDB_API_KEY=<YOUR_KEY> \
     sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
     ```

   **Hint**: Instead of inline environment variables, you can **export** them in advance:
   ```bash
   export WANDB_API_KEY=<YOUR_KEY>
   export HTTP_PROXY="http://125.209.240.12:10038"
   export HTTPS_PROXY="http://125.209.240.12:10038"

   sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
   ```
*If your shell is configured with `HISTCONTROL=ignorespace`, you can prefix these commands with a space (` export WANDB_API_KEY=xxx`) to skip logging them in your shell history. This is useful to keep sensitive values private.*

### **Resuming from a Specific Checkpoint**
If you want to resume training from a particular checkpoint (e.g., `checkpoint-500`):
- Update your config YAML to include:
  ```yaml
  resume_from_checkpoint: checkpoint-500
  ```
  or
- Pass it as an environment variable or command-line argument (depending on how your setup reads it).  
  This ensures that your training picks up exactly where you left off, reloading the model weights and continuing the training steps from that checkpoint.

7. **Monitor Logs**
    - The main Slurm logs are in `logs/%j-%x-sbatch.out/.err` (relative to the **project root**).
    - **Additionally**, per-node `srun` logs are written in `logs/%j-%x-%N-srun.out`/`.err`.
    - You can easily monitor the logs by using wildcards in `tail -f`: `tail -f logs/%j-*` (watches all logs produced by the current job id `%j`).

8. **Check W&B and Outputs**
    - If `report_to` includes `wandb` in your YAML, logs appear on wandb.ai using your `WANDB_API_KEY`.
    - **WandB Forking**: If you also want to “fork” from a rewound run at a specific step, set `resume_from_checkpoint` and `WANDB_FORK_FROM="abc123?_step=500"`, etc.
    - Model artifacts and checkpoints are stored in the `output_dir` specified in your YAML (e.g., `/mnt/clovanap/audiollm/results`).

6. **Submit the Slurm Job**  
   - Provide your Weights & Biases API key, and optionally the proxy variables, then call `sbatch` against the script in `spd2/`:
     ```bash
     WANDB_API_KEY=<YOUR_KEY> \
     HTTP_PROXY="http://125.209.240.12:10038" \
     HTTPS_PROXY="http://125.209.240.12:10038" \
     sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
     ```
   - If no proxy is needed:
     ```bash
     WANDB_API_KEY=<YOUR_KEY> \
     sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
     ```

   *Instead of inline environment variables, you can **export** them in advance:*
   ```bash
   export WANDB_API_KEY=<YOUR_KEY>
   export HTTP_PROXY="http://125.209.240.12:10038"
   export HTTPS_PROXY="http://125.209.240.12:10038"

   sbatch spd2/audiollm-speechx.sbatch configs/speechx-v6/speechx-v6-s2-spd2.yaml
   ```
*(If your shell is configured with `HISTCONTROL=ignorespace`, you can prefix these commands with a space (` export WANDB_API_KEY=xxx`) to avoid storing them in your shell history. This is useful for keeping sensitive values private.)*

- **Resuming from a specific checkpoint**:  
  If you wish to continue training from a certain checkpoint (e.g., `checkpoint-500`), update your YAML to include:
  ```yaml
  resume_from_checkpoint: checkpoint-500
  ```
  Alternatively, pass that argument or environment variable in whatever way your setup reads it. This ensures your training restarts at the correct step, reloading the model weights and continuing seamlessly from the previous run.

7. **Monitor Logs**
    - The main Slurm logs are in `logs/%j-%x-sbatch.out/.err` (relative to the **project root**).
    - **Additionally**, per-node `srun` logs are written in `logs/%j-%x-%N-srun.out`/`.err`.
    - You can easily monitor the logs by using wildcards in `tail -f`: `tail -f logs/%j-*` (watches all logs produced by the current job id `%j`).

8. **Check W&B and Outputs**
    - If `report_to` includes `wandb` in your YAML, logs appear on wandb.ai using your `WANDB_API_KEY`.
    - **WandB Forking**: If you also want to “fork” from a rewound run at a specific step, set both `resume_from_checkpoint` (e.g., `checkpoint-500`) and an environment variable like `WANDB_FORK_FROM="abc123?_step=500"` to branch from that run and step.
    - Model artifacts and checkpoints are stored in the `output_dir` specified in your YAML (e.g., `/mnt/clovanap/audiollm/results`).

---

## Notes

- **CPU/Memory Binding**:  
  Adjust `bind.sh` or the parameters in the final `srun` command if you need a different binding policy (e.g., `--cpu=exclusive`).

- **Mount Paths**:  
  The batch script uses `--container-mounts="/mnt:/mnt,/home:/home"`; modify if your environment demands additional or fewer mounts.

- **`run.sh` Usage (Optional)**:  
  You can also test or run locally via `run.sh`, which calls `torchrun`:
  ```bash
  # Syntax:
  bash spd2/run.sh [CONFIG_FILE]
  
  # Example:
  bash spd2/run.sh configs/speechx-v6-s2-spd2.yaml
  ```
  If no `CONFIG_FILE` is given, it defaults to `configs/speechx-v6-s2-spd2.yaml`. This approach does **not** rely on Slurm or the container environment.

- **Container Building**:  
  If you change `Dockerfile`, re-build (`docker build ...`) and push to your registry or convert to `.sqsh` for Enroot usage.

By following these steps **from the project root directory**, you can reliably train your Speech X models on SuperPOD 2 using the recommended SPD2 config found in `configs/speechx-v6/speechx-v6-s2-spd2.yaml`.