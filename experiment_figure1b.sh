#!/bin/bash

cd Code
algos=("LinGame" "LinGIFA" "LinGapE" "LUCB" "UGapE")
for algo in "${algos[@]}"; do
	if [ "$algo" == "LinGame" ]; then
		## "Quick" threshold function in log
		beta="Heuristic"
	else
		if [[ "$algo" == "LUCB" || "$algo" == "UGapE" ]]; then
			## Threshold function from Kalyanakrishnan and colleagues' original paper on LUCB
			beta="LUCB1"
		else
			## Abbasi-Yadkori and colleagues' threshold function
			beta="Frequentist"
		fi
	fi
	cmd="python main.py --small_K 3 --beta "$beta" --data classic --omega 0.1 --sigma 0.5 --bandit "$algo" --is_greedy 0 --m 1 --n_simu 500 --epsilon 0. --delta 0.05 --json_file ../Results/classic_0.1_K3_gaussian_m=1_delta=0.05_epsilon=0.0/parameters.json"
	echo $cmd
	$cmd
done
