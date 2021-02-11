#!/bin/bash

cd Code
algos=("LinGapE" "LinGIFA" "LUCB")
greedy=("0" "1")
for algo in "${algos[@]}"; do
	for gr in "${greedy[@]}"; do
		cmd="python main.py --small_K 10 --beta Heuristic --data epilepsy --sigma 0.5 --bandit "$algo" --is_greedy "$gr" --m 5 --n_simu 500 --epsilon 0. --delta 0.05 --json_file ../Results/epilepsy_epilepsy_m=5_delta=0.05_epsilon=0.0/parameters.json"
		echo $cmd
		$cmd
	done
done
