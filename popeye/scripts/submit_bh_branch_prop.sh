#!/bin/bash
# Standard output and error:
#SBATCH -o ./bh_tab.out.%j
#SBATCH -e ./bh_tab.err.%j
# Initial working directory
#SBATCH -D /home/cduckworth/bh_star_gas_misalignment/popeye/scripts
# Job name:
#SBATCH -J "BH props MPL-8"
# Number of nodes and threads.
#SBATCH -N1 --exclusive
#
# Wall clock limit.
#SBATCH --time=48:00:00

# Run the program.
module load slurm python3 gcc
python3 ./compute_bh_branch_properties.py
