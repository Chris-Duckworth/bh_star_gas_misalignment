#!/bin/bash
# Standard output and error:
#SBATCH -o ./aniso_out.%j
#SBATCH -e ./aniso_err.%j
# Initial working directory
#SBATCH -D /home/cduckworth/bh_star_gas_misalignment/popeye/scripts
# Job name:
#SBATCH -J "anisotropy MPL-8"
# Number of nodes and threads.
#SBATCH -N1 --exclusive
#
# Wall clock limit.
#SBATCH --time=05:00:00

# Run the program.
module load slurm python3 gcc
python3 ./compute_anisotropy_radii.py
