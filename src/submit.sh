#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=denoise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=denoise.out
#SBATCH --exclusive
#SBATCH --gres=gpu:tesla_k20m:1

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."


#Strong scaling
echo "0: serial code"
./noise_remover -i coffee.pgm -iter 200 -o serial.png

echo "1: part 1 code"
./part1 -i coffee.pgm -iter 200 -o part1.png

echo "2: part 2 code"
./part2 -i coffee.pgm -iter 200 -o part2.png

echo "3: part 3 code"
./part3 -i coffee.pgm -iter 200 -o part3.png