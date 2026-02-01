#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --account=cs6969
#SBATCH --cpus-per-task=96
#SBATCH -t 0:20:00
#SBATCH --export=SPP,SCATTERS,SAMPLEPROB

# Usage: sbatch --export=SPP=1024,SCATTERS=64,SAMPLEPROB=0.9 lambda.bat

LOGFILE=$PWD/lambda.$SLURMD_NODENAME.$SLURM_JOB_ID.log

echo "*** Assigned Granite Node: " $SLURMD_NODENAME \
  | tee $LOGFILE

# load g++-15
module load gcc/15.1.0

echo "*** Compiling..." >> $LOGFILE
make release

cd bin
echo "*** Rendering..." >> $LOGFILE
(time ./lambda scenes/dispersion.nls -s $SPP -b $SCATTERS -p $SAMPLEPROB) \
  &>> $LOGFILE