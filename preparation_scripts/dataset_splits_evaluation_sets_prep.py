import argparse
import pandas as pd
import string
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="add the path to your cv corpus directory", required=True)
args = parser.parse_args()

validated = pd.read_csv(args.data + '/sw/validated.tsv', sep='\t', low_memory=False)

# Text normalization function
# 1. we lower all letters
# 2. we get rid of leading and trailing spaces
# 3. we get rid of extra spaces
# 4. remove punctuation
# 5. dropping all the sentences which have digits, these look quite problematic!!
# 6. remove all characters not in my 'allowed' list

def normalize_text(df):
    df.loc[:,'sentence'] = df['sentence'].apply(lambda s: s.lower())
    df.loc[:,'sentence'] = df['sentence'].apply(lambda s: s.strip())
    df.loc[:,'sentence'] = df['sentence'].apply(lambda s: " ".join(s.split()))
    df.loc[:,'sentence'] = df['sentence'].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
    df = df[~df['sentence'].str.contains('\d')]
    characters_allowed = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
                          'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',]
    df.loc[:,'sentence'] = df['sentence'].apply(lambda s: ''.join(char for char in s if char in characters_allowed or char == ' '))

    return df

# filtering out data from the following PRs. These are either domain data or dialect and variant data. These subsets are intended for fine-tuning and evaluation respectively.
# so far only the variant/dialect sets have been uploaded 
# https://github.com/common-voice/common-voice/pull/3957 by Badili innovations, livestock data
# https://github.com/common-voice/common-voice/pull/3970 by Badili innovations, livestock data
# https://github.com/common-voice/common-voice/pull/3640 by Kat, Dialect/Variant data

kiunguja = pd.read_csv('experiment_data/sw-kiunguja.txt', header=None)
baratz = pd.read_csv('experiment_data/sw-baratz.txt', header=None)
kibajuni = pd.read_csv('experiment_data/sw-kibajuni.txt', header=None)
kimakunduchi = pd.read_csv('experiment_data/sw-kimakunduchi.txt', header=None)
kimvita = pd.read_csv('experiment_data/sw-kimvita.txt', header=None)
kipemba = pd.read_csv('experiment_data/sw-kipemba.txt', header=None)
kitumbatu = pd.read_csv('experiment_data/sw-kitumbatu.txt', header=None)

# function to save our dialect/variant evaluation sets, we want to save these in alignment with the rest of our subsets
def dialect_evaluation_sets(df):
    dialect_set = []
    for idx, value in validated['sentence'].items():
        for idx2, value2 in df[0].items():
            if value == value2:
                dialect_set.append(idx)

    df = validated[validated.index.isin(dialect_set)]

    return normalize_text(df)

dialect_evaluation_sets(kiunguja).to_csv('experiment_data/eval_kiunguja.csv', index=False)
dialect_evaluation_sets(kibajuni).to_csv('experiment_data/eval_kibajuni.csv', index=False)
dialect_evaluation_sets(baratz).to_csv('experiment_data/eval_baratz.csv', index=False)
dialect_evaluation_sets(kimakunduchi).to_csv('experiment_data/eval_kimakunduchi.csv', index=False)
dialect_evaluation_sets(kimvita).to_csv('experiment_data/eval_kimvita.csv', index=False)
dialect_evaluation_sets(kipemba).to_csv('experiment_data/eval_kipemba.csv', index=False)
dialect_evaluation_sets(kitumbatu).to_csv('experiment_data/eval_kitumbatu.csv', index=False)

#we then filter them out of the df that we continue working with
filter_out = pd.concat([kiunguja[0],baratz[0],kibajuni[0],kimakunduchi[0],kimvita[0],kipemba[0],kitumbatu[0]], ignore_index=True)
filter_out = pd.DataFrame(filter_out)

# finding the instances from the filter_out set
filter_sets = []
for idx, value in validated['sentence'].items():
    for idx2, value2 in filter_out[0].items():
        if value == value2:
            filter_sets.append(idx)

# filtering out the instances in out filter set
filtered_validated = validated[~validated.index.isin(filter_sets)]
filtered_validated = normalize_text(filtered_validated)

# considerations in creating the train, dev, test sets
# repeated instances of sentences should only be in one set
# repeated audios contributed by a single speaker should also all only be in one set
# balancing gender in the various sets
# we also drop duplicates where a user may have contributed to an individual sentence more than once
# we split out data into 4 subsets with the following ratio; 60:15:15:10
# the final 10% will create evaluation sets to help us quantify gender and age bias

def split_data(df):
    # Group by sentence and client_id
    grouped = df.groupby(['sentence', 'client_id'], as_index=False)

    # Get the first index of each group
    idxs = grouped.first().index

    # Split indices into train, validation, and test sets
    train_idxs, val_test_idxs = train_test_split(idxs, test_size=0.4, random_state=42)
    val_idxs, test_idxs = train_test_split(val_test_idxs, test_size=0.75, random_state=42)
    val2_idxs, test2_idxs = train_test_split(test_idxs, test_size=0.5, random_state=42)

    # Get the corresponding sentences and client_ids for each set of indices
    train = df.loc[df.index.isin(train_idxs)]
    val = df.loc[df.index.isin(val_idxs)]
    val2 = df.loc[df.index.isin(val2_idxs)]
    test = df.loc[df.index.isin(test2_idxs)]

    return train, val, val2, test

train, eval0, dev, test = split_data(filtered_validated)

# save my train, dev and test sets
train.to_csv('experiment_data/train.csv', index=False)
dev.to_csv('experiment_data/dev.csv', index=False)
test.to_csv('experiment_data/test.csv', index=False)

# save my age and gender evaluation sets
eval0[eval0['gender'] == 'male'].head(5000).to_csv('experiment_data/eval_male.csv', index=False)
eval0[eval0['gender'] == 'female'].head(5000).to_csv('experiment_data/eval_female.csv', index=False)
eval0[eval0['age'] == 'twenties'].head(5000).to_csv('experiment_data/eval_20s.csv', index=False)
pd.concat([eval0[eval0['age'] == 'thirties'].head(2719), eval0[eval0['age'] == 'fourties'], eval0[eval0['age'] == 'fifties'], eval0[eval0['age'] == 'sixties']]).to_csv('experiment_data/eval_o30s.csv', index=False)

# save my age x gender evaluation sets
eval0.loc[(eval0['gender'] == 'male') & (eval0['age'] == 'twenties')].to_csv('experiment_data/eval_20sMale.csv', index=False)
eval0.loc[(eval0['gender'] == 'female') & (eval0['age'] == 'twenties')].to_csv('experiment_data/eval_20sFemale.csv', index=False)
eval0.loc[(eval0['gender'] == 'female') & ~(eval0['age'] == 'twenties') & ~(eval0['age'] == 'teens')].to_csv('experiment_data/eval_o30sFemale.csv', index=False)
eval0.loc[(eval0['gender'] == 'male') & ~(eval0['age'] == 'twenties') & ~(eval0['age'] == 'teens')].to_csv('experiment_data/eval_o30sMale.csv', index=False)
