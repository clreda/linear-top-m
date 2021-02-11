# linear-top-m

Code associated with AISTATS 2021 paper "Top-m identification for linear bandits". The code was successfully run on **Python 2.7.14+**, on Linux Debian 8. To cite the paper, please use:

```
@inproceedings{reda2021top,
  title={Top-m identification for linear bandits},
  author={R\'{e}da, Cl\'{e}mence and Kaufmann, \'{E}milie and Delahaye-Duriez, Andr\'{e}e},
  booktitle={Proceedings of the $24^{th}$ International Conference on Artificial Intelligence and Statistics (AISTATS) 2021},
  volume={130},
  year={2021}
}
```

## TL;DR

+ Install necessary packages:

```bash
python -m pip install --force-reinstall pip==9.0.1
pip install -r requirements.txt
//most up-to-date version of https://github.com/regulomics/expansion-network according to GitHub
git clone https://github.com/clreda/expansion-network expansion-network
```

+ Data for the instance of drug repurposing for epilepsy (concept id/CID: C0014544) is stored in **DR_Data/** (data is from LINCS L1000 and RepoDB, and data from GEO accession number GSE77578).

+ In order to reproduce the exact figures shown in the paper, run generate\_pdf.py which uses files in folder *Results/*.

## Command options

- default argument values are written in JSON in file *args.json*. Necessary arguments are:

	+ **small\_N** for **data**=linear or logistic
	+ **small\_K** for **data**=classic or linear or logistic
	+ **vr** for **data**=linear or logistic
	+ **omega** for **data**=classic
	+ **T\_init** for **bandit**=TrueUniform
	+ **bandit**
	+ **data**
	+ **m**
	+ **beta**

- Adding "--verbose 1" allows to display comments and description of selected arms and obtained rewards.

## Modifying the code

### Add a new bandit algorithm

- Check file *bandits.py*. You need to create a sub-instance of **ExploreMBandit** and add it to the *bandit\_factory*. Then you will be able to invoke it from the terminal with option "--bandit".

### Add a new type of data

- Check file *data.py*. You need to modify function *create\_scores\_features*. Then you will be able to invoke it from the terminal with option "--data".

### Add a new threshold function

- Check file *betas.py*. You need to create a function similar to the **AlphaDependentBeta**, **LUCB1Beta**, ... and add it to the *beta\_factory*. Then you will be able to invoke it from the terminal with option "--beta".

### Add a new problem type

- Check file *problems.py*. You need to create a sub-instance of **GenericProblem** and add it to the *problem\_factory*. Then you will be able to invoke it from the terminal with option "--problem".
