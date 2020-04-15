#!/bin/bash
#$ -pe omp 12
#$ -l h_rt=48:00:00

####################################################################
# Script to submit the Markov Chain Stabilizer as GridEngine batch
# job. stabilize.py cannot find the Hidden Markov Model (hmm) module
# when submitted to the grid system directly.
####################################################################

time ./stabilize.py $*
