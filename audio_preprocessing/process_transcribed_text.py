"""
Takes a file with transcribed audio from an ASR system, transforms it into the format that the original VQA has
parsed its questions, and puts it into an hdf5 file. TextMod can then take inputs from this file. 

The file containing audio transcriptions is expected to have the format question_id:question string like shown below
10,WHAT SERVICE DOES THE CAR PARK THE CAR PROVIDE 
1000070,IS THIS GIRAFFE AND ASSUME 
1000071,ARE THE LUMPS ON THE GROUND ROCKS 
1000072,WHAT IS THAT ANIMAL 
...
"""

import h5py
import json
from nltk import word_tokenize
import numpy as np

someDict = {}
transcripFile = 'path_to_transcribed_text.txt'
with open(transcripFile, 'r') as tf:
    for line in tf:
        splits = line.strip().split(',')
        someDict[splits[0]] = splits[1]

# the dictionary contains "str(question id):str(question)"

# change these paths and variables for different datasets in train/val/test
original_questionfile = h5py.File('somepath/data_prepro.h5') # this file contains the questions as input form
question_idxfile = json.load(open('somepath/data_prepro.json','r'))
qids = original_questionfile.get('question_id_test') # or train or whatever
resultArrShape = original_questionfile.get('ques_test').shape

idx_to_word = question_idxfile['ix_to_word']
word_to_idx = {}
for k in idx_to_word.keys():
    word = idx_to_word[k]
    word_to_idx[word] = k

outputArr = np.zeros(resultArrShape, dtype='uint16')

for n, id in enumerate(qids):
    trans_str = someDict[str(id)]
    # we expect all the ids to be parsed and exist
    allWords = word_tokenize(trans_str.lower())
    allWords.append('?')
    for i, word in enumerate(allWords):
        try:
            outputArr[n,i] = int(word_to_idx[word])
            # use the original index number
        except KeyError:
            outputArr[n,i] = 0
            # can't find the word, use 0, which is masked

# write the result to an h5 file
# change these paths and variables for different datasets in train/val/test
output = h5py.File('transcribed_questions.h5')
output.create_dataset('ques_test', data=outputArr)
output.close()