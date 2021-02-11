#coding: utf-8

import pandas as pd
import numpy as np
import sys
from glob import glob
import subprocess as sb

pwd = sb.check_output("pwd", shell=True)
sys.path.insert(1, "/".join(pwd.split("/")[:-1])+'/Code/')
from utils import capture, capture_output

def read_csv(filen):
	S = pd.read_csv(filen, header=0)
	S.index = S[S.columns[0]]
	S = S.drop(columns=[S.columns[0]])
	S = S[~S.index.duplicated()]
	return S

## disease name
disease_name = "Epilepsy"
## number of samples
n_samples = len(pd.read_csv("GSE77578_patients.csv", header=0).columns)-1
## Binary drug signatures (for our scoring function)
S = read_csv(disease_name.lower()+"_signatures_binarized.csv")
## Non-binary drug signatures (for the baseline method)
X = read_csv(disease_name.lower()+"_signatures_nonbinarized.csv")

assert sys.argv[1] in ["adj_cosine", "cosine", "tanimoto"]
try:
	n_drugs_max = int(sys.argv[2])
except:
	if (sys.argv[2] == "subset"):
		n_drugs_max = "subset"
	else:
		n_drugs_max = "all"
sample_begin, sample_end = int(sys.argv[3]) if (len(sys.argv) >= 4) else 1, int(sys.argv[4]) if (len(sys.argv) == 5) else n_samples
verified = True
save = True

### ARGUMENTS for bandit instance
## Get scores (ordered by PubChem CID in signatures)
score_file = pd.read_csv(disease_name.lower()+"_scores.csv")

print(score_file["score"].value_counts())

score_file.index = list(score_file[score_file.columns[0]])
## Get signatures (ordered by PubChem CID in signatures)
S = S[X.columns]
scores = score_file.loc[list(map(int,X.columns))]["score"].values.flatten().tolist()
## Get names (ordered by PubChem CID in signatures)
names = score_file.loc[list(map(int,X.columns))]["drug_name"].values.flatten().tolist()
## Get GRN
grn_name_path = glob("GRN/*solution.net")
assert len(grn_name_path) == 1
grn_name_path = grn_name_path[0]
grn_name = grn_name_path.split("/")[-1]
path_to_grn = "../expansion-network/examples/models/epilepsy/"

assert len(scores) == len(S.columns)
## Select a subset of drugs
drug_1 = (np.argwhere(np.array(scores) == 1)).flatten().tolist()
drug_0 = (np.argwhere(np.array(scores) == 0)).flatten().tolist()
drug_m1 = (np.argwhere(np.array(scores) == -1)).flatten().tolist()
if (n_drugs_max == "subset"):
	## Known anti-epileptics
	selected_names = ["Hydroxyzine", "Acetazolamide", "Pentobarbital", "Topiramate", "Diazepam"]
	## Known pro-convulsants
	selected_names += ["Dmcm", "Brucine", "Fipronil", "Flumazenil", "Fg-7142"]
	drug_interval = [names.index(drug) for drug in selected_names]
else:
	if (n_drugs_max != "all" and n_drugs_max > 0):
		from random import seed, sample
		seed(0)
		drug_interval = reduce(lambda x,y : x+y, [sample(drugs, min(n_drugs_max, len(drugs))) for drugs in [drug_1,drug_0,drug_m1]])
	elif (n_drugs_max != "all"):
		drug_interval = drug_1+drug_m1
	else:
		assert n_drugs_max == "all"
		drug_interval = range(len(scores))

print("Number of tested drugs = "+str(len(drug_interval)))
samples = range(sample_begin-1, sample_end)
print("Number of tested samples = "+str(len(samples)))

fname = "rewards_"+sys.argv[1]+"_"+str(len(drug_interval))+"drugs_"+(str(len(samples)) if (len(samples) == n_samples) else str(sample_begin)+"-"+str(sample_end)+"_")+"samples.txt"

## Run tests
from problems import DRProblem
problem = DRProblem(scores, "epilepsy", {"names": names, "X": X, "S": S, "grn_name": grn_name_path, "path_to_grn": path_to_grn}, func=sys.argv[1], quiet=False)
rewards = np.zeros((len(samples), len(drug_interval)))
for sample_id in range(sample_begin-1, sample_end):
	print("\n\nSample id: "+str(sample_id+1)+" ("+str(samples.index(sample_id))+"/"+str(len(samples))+")")
	for sa, arm in enumerate(drug_interval):
		with capture() as out:
			reward = problem.get_reward(arm, quiet=False, sample_id=sample_id)
		print(out)
		with open(fname, "a+") as f:
			f.write(str(out))
		rewards[samples.index(sample_id), sa] = reward

## Save results
if (save):
	import pickle
	rewards = pd.DataFrame(rewards, columns=[names[i] for i in drug_interval], index=pd.read_csv(patients_file, header=0).columns[[s+1 for s in samples]])
	rewards.loc["score"] = [scores[i] for i in drug_interval]
	rewards.loc["drug_id"] = drug_interval
	rewards.to_csv(fname.split(".txt")[0]+".csv")

## Cleaning
sb.call("rm -f ../Code/*.pyc", shell=True)
