#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=96
#SBATCH -t 0:10:00
#SBATCH --export=ALL,SPP,SCATTERS,SAMPLEPROB

# Usage: sbatch -v SPP=1024,SCATTERS=64,SAMPLEPROB=0.9 lambda.bat 

echo "*** Assigned Granite Node: " $SLURMD_NODENAME \
  | tee lambda.$SLURMD_NODENAME.$SLURM_JOB_ID.log

# load g++-15
module load gcc/15.1.0

# compile renderer
make release

# run renderer
cd bin
time ./lambda scenes/teapot.nls -s $SPP -b $SCATTERS -p $SAMPLEPROB \
  >> lambda.$SLURMD_NODENAME.$SLURM_JOB_ID.log 2>&1
