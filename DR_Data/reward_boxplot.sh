#!/bin/bash

pip install -r requirements.txt
pip install python-igraph cairocffi bs4 ipython
## /!\ a little time-consuming
# we want to run the simulator on all patient samples on the preselected subset of drugs
python generate_rewards.py cosine subset
# generate boxplot from the generated rewards on the subset of 10 drugs on 18 patients
python parse_rewards.py cosine 18 10

