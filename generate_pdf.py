# coding:utf-8

from glob import glob
import subprocess as sb
import os

with open("Code/args.json", "r") as f:
	args = f.read()
with open("args_latex.json", "w") as f:
	str_ = args.split("small_test")
	f.write(str_[0]+"generate_latex"+str_[1])

gen_files = glob("./Results/*/last_command.txt")
for c in gen_files:
	with open(c, "r") as f:
		x = f.read()
		y = "--json_file ../args_latex.json"
		cmd = "cd Code; "+x
		if (not y in x):
			cmd += " "+y
		cmd += "; cd .."
		print(cmd)
		sb.call(cmd, shell=True)

pdf_files = glob("./Results/*/*.pdf")
if (not os.path.exists("pdfs/")):
	sb.call("mkdir pdfs/", shell=True)
for c in pdf_files:
	sb.call("mv "+c+" pdfs/", shell=True)
	sb.call("mv "+c[:-4]+".tex pdfs/", shell=True)
	sb.call("mv "+c[:-4]+".png pdfs/", shell=True)

sb.call("rm -rf args_latex.json", shell=True)
