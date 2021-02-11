#!/bin/bash

cd Code
## LinIAA = LinGIFA with individual indices
## LinGIFAPlus = LinGIFA with tau^LUCB stopping rule
algos=("LinGapE" "LinGIFA" "LinIAA" "LinGIFAPlus" "LUCB" "UGapE")
for algo in "${algos[@]}"; do
	echo $algo
	## Abbasi-Yadkori and colleagues' threshold function
	beta="Frequentist"
	if [[ "$algo" == "LinGIFA" || "$algo" == "LinGapE" || "$algo" == "LinGIFAPlus" ]]; then
		greedy=("0" "1")
	else
		if [[ "$algo" == "LinIAA" ]]; then
			greedy=("0")
		else
			greedy=("1")
			## Threshold function from Kalyanakrishnan and colleagues' original paper on LUCB
			beta="LUCB1"
		fi
	fi
	for gr in "${greedy[@]}"; do
		cmd="python main.py --small_K 4 --beta "$beta" --data classic --omega pi/3 --sigma 0.5 --bandit "$algo" --is_greedy "$gr" --m 2 --n_simu 500 --epsilon 0. --delta 0.05 --json_file ../Results/classic_pi3_K4_gaussian_m=2_delta=0.05_epsilon=0.0/parameters.json"
		echo $cmd
		$cmd
	done
done
