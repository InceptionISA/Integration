


# this file must merge the latest submission with the latest submission of the submissions/face, submissions/track
# it find the latest by looking at the timestamp of the submission file
# the files named submission_{i}_*.csv

# then merge the two files. using this logic:


import pandas as pd
tracking_data = pd.read_csv('tracking/submission_4_*.csv')
tracking_data
reid = pd.read_csv('reid/submission_5_*.csv')

two files must be this columns 

id	frame	objects	objective


merged_df = pd.concat([tracking_data, reid], ignore_index=True)

it must observe the last index of the submisions and increment it by 1

merged_df
sub_file = merged_df.drop(columns='id').reset_index()
sub_file.rename(columns={'index': 'ID'}, inplace=True)

submission_file['objects'] = submission_file['objects'].apply(
    lambda x: ast.literal_eval(x))

# Assert that the first object's data type is a list
assert isinstance(submission_file['objects'].iloc[0],
                  list), "The first 'objects' entry is not a list!"

sub_file.to_csv('submissions/submission_{i}_.csv', index=False)



# then it will submit the file into the competition using this name of competition:

surveillance-for -retail-stores



then retrive the results from the competition and print the results

all via kaggle api, but you will want the kaggle.json
you can find it in the root directory of the project ( same directory as this file) 


then you will create expiremnets/experiment_{latest number}_public_score in experimnets directory.

and store in it the following:

{
    "submission": "submissions/submission_{i}_.csv",
    "public_score": 0.5
    "trackfile": "submissions/Track/submission_{i}_.csv",
    "facefile": "sumbissions/Face/submission_{i}_.csv",
}