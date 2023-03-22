import itertools
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="add the path to the csv for the training data subset", required=True)
args = parser.parse_args()

train = pd.read_csv(args.train)

# dictionary where we will track sentences and their duplicates
duplicates = {}

# loop through every unique sentence in the  training set
for sentence in train['sentence'].unique():

    # get the indices of the rows where this sentence occurs
    indices = train.index[train['sentence'] == sentence].tolist()

    # add the sentence and its corresponding indices to the dictionary
    duplicates[sentence] = indices

# we create the experiment training sets with the repeated instances of each item 
one2one = []
one2two = []
one2four = []
one2eight = []

for dups in duplicates.values():
    one2one.append(dups[0:1])
    one2two.append(dups[0:2])
    one2four.append(dups[0:4])
    one2eight.append(dups[0:8])

one2one = [item for sublist in one2one for item in sublist]
one2two = [item for sublist in one2two for item in sublist]
one2four = [item for sublist in one2four for item in sublist]
one2eight = [item for sublist in one2eight for item in sublist]

# save csv files of our experiment training sets
train.iloc[one2one].to_csv('experiment_data/one2one_train.csv', index=True)
train.iloc[one2two].to_csv('experiment_data/one2two_train.csv', index=True)
train.iloc[one2four].to_csv('experiment_data/one2four_train.csv', index=True)
train.iloc[one2eight].to_csv('experiment_data/one2eight_train.csv', index=True)
