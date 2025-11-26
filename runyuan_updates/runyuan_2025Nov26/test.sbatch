#!/bin/bash
#SBATCH -J testmulti
#SBATCH -p mi2104x
#SBATCH -A mzhang/runyuan15
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=00:30:00

echo "HELLO FROM JOB"
