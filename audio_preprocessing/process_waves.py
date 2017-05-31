"""
This file reads a bunch of .mp3 files and puts the waveform into one giant hdf5 file.
To read .wav files, just change the 'fileStr' variable to have .wav extension instead
"""
import h5py
import numpy as np
import os
import time
from moviepy.editor import AudioFileClip

# path to directory containing audio files
audioFileDir = 'some/path/val2014_audioFiles'

hf = h5py.File('../data/data_prepro.h5')
quesID = hf.get('question_id_test') # if you want train, then use the corresponding key

# create the output file
outputFileName = 'waveform_val'
hf = h5py.File(outputFileName, 'w')
dt = h5py.special_dtype(vlen=np.dtype('float32'))
dset = hf.create_dataset('val', shape=(len(quesID), ), dtype=dt)

output = []
begin = 0
s_t = time.time()
for n, i in enumerate(quesID):
    # loop through all the question IDs, and find the corresponding audio files in the audio files directory
    fileStr = '{}/{}.mp3'.format(audioFileDir, i)
    if not os.path.exists(fileStr):
        print "Cannot find + " + fileStr
        raise ValueError
    try:
        audio = AudioFileClip(fileStr, fps=16000) # could be a wav, mp3...
    except IndexError as e:
        print "Could not read audio file {}".format(fileStr)
        raise e
    s = audio.to_soundarray()[:,0]
    output.append(s.astype('float32'))
    # write to the dataset after processing 1000 samples, to reduce I/O operations
    writechunks = 1000
    somenum = (n+1) % writechunks
    if somenum == 0:
        dset[begin:n+1] = np.array(output)
        begin = n+1
        output = []
        t_t = time.time()
        print "Finished {}, time taken: {}".format(n+1, t_t-s_t)
        s_t = time.time()

dset[begin:] = output
hf.close()
