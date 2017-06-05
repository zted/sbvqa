"""
Takes a file with transcribed audio from an ASR system, and calculates WER based on comparison with original questions.
The file is expected to have the format question_id:question string like shown below
10,WHAT SERVICE DOES THE CAR PARK THE CAR PROVIDE 
1000070,IS THIS GIRAFFE AND ASSUME 
1000071,ARE THE LUMPS ON THE GROUND ROCKS 
1000072,WHAT IS THAT ANIMAL 
...
"""

from nltk.tokenize import word_tokenize
import json


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    
    Sourced from Martin Thoma (https://martin-thoma.com/word-error-rate-calculation/)
    
    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

# parse transcription
someDict = {}
transcripFile = 'path_to_transcribed_text.txt'
with open(transcripFile, 'r') as tf:
    for line in tf:
        splits = line.strip().split(',')
        someDict[splits[0]] = splits[1]

j = json.load(open('somePath/OpenEnded_mscoco_test-val2014_questions.json','r'))
wertot = 0
matchSymbols = "([-.\"',:?!\$#@~()*&\^%;\[\]/\\\+<>\n=])"

for question in j['questions']:
    quesStr = str(question['question'])
    # gets the question from json
    quesID = str(question['question_id'])
    s1 = word_tokenize(quesStr.lower())
    # removes punctuation
    s1 = [s for s in s1 if s not in matchSymbols]
    s2 = word_tokenize(someDict[quesID].lower())
    # tokenizes from transcribed file
    errorPercent = wer(s1, s2)/float(len(s1))
    wertot += errorPercent
print wertot/len(j['questions'])
