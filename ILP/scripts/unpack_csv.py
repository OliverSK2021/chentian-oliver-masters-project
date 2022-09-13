import pandas as pd
import sys
import os

csv_file = sys.argv[1]
prefix = sys.argv[2]

try:
	os.mkdir(prefix + "_positive")
except FileExistsError:
	print("Positive folder already exists")
try:
	os.mkdir(prefix + "_negative")
except FileExistsError:
	print("Negative folder already exists")

df = pd.read_csv(csv_file)

def strip_cruft(code):
	start = code.find("/n/n") + 4
	finish = code.find("/n/n/n")
	return(code[start:finish])

for i in range(len(df)):
	if df["label"][i] == 0:
		f = open(prefix + "_negative/" + df["id"][i] + ".py", "w")
	else:
		f = open(prefix + "_positive/" + df["id"][i] + ".py", "w")
	f.write(strip_cruft(df["code"][i]))
	f.close()