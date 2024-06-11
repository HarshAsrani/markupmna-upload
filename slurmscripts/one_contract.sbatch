#!/bin/bash
#SBATCH --job-name=run_nli       # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=12        # Number of CPU cores per task
#SBATCH --mem=64GB               # Memory per node
#SBATCH --time=8:00:00          # Time limit hrs:min:sec
#SBATCH --gres=gpu:1              # Number of GPUs
# #SBATCH --partition=gpu         # GPU partition
# #SBATCH --account=your_account  # Your account name
#SBATCH --output=op1contract.txt  # Output file
#SBATCH --error=err1contract.txt    # Error file

# Load necessary modules and activate Conda environment
# module load cuda/your_cuda_version  # Specify the CUDA version
# module load anaconda/your_anaconda_version  # Specify the Anaconda version
conda init bash  # Assuming your shell is Bash
source ~/.bashrc  # Update the shell environment
conda activate indenv

# Navigate to the directory containing your script
cd /scratch/ha2423/multiclass/multiclass/contract-nli-bert

# Run the shell script
python train.py ./data/conf_base.yml ./output_job

