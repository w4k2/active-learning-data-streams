#! /bin/bash

./run.sh "LeastConfidence" "cuda:1" &> least_conf.log &
./run.sh "MarginSampling" "cuda:2" &> least_conf.log &
./run.sh "EntropySampling" "cuda:3" &> entropy_sampl.log &
./run.sh "BALDDropout"  "cuda:4" &> bald_drop.log &
./run.sh "ConsensusEntropy" "cuda:5" &> consensus_entrop.log &
./run.sh  "Ours" "cuda:6" &> ours.log &