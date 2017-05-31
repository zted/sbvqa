"""
Corrupts the original audio with noise
1. change the and output paths, at the moment original audio is assumed to be mp3 format, noise is wav format
2. make sure the output folders exist. if we're corrupting at noise levels of 15 on validation set,
 the output folder somepath/val_noisy15 needs to exist
"""

import glob
import os
import time
import wave
from math import ceil
from random import randint

import numpy as np
from pydub import AudioSegment

noise_levels = [10, 20, 30, 40, 50]  # noise levels to corrupt at. 10 means 10% noise, 90% original audio
sample_rate = 16000

dataSet = 'val'

# path to original audio files
original_audio_path = '/somepath/audiofiles_val/'
# path to files containing noise
path_noise = '/somepath/UrbanSound8K/'
# path to store the output files
output_path = '/somepath'

# log exactly which noise file is combined with which original audio file
logfile = './{}_noisy.txt'.format(dataSet)
flog = open(logfile, 'w')

noise_files = [f for f in glob.glob(path_noise + "*.wav")]
speech_files = [f for f in glob.glob(original_audio_path + "*.mp3")]

assert len(noise_files) == len(speech_files)
# all noisy files
writeStr = ''
prev_time = time.time()
for i, speechFilePath in enumerate(speech_files):
    # ingest original audio file
    [_, speechFileName] = os.path.split(speechFilePath)
    speechID = speechFileName.split('.')[0]
    try:
        speech_audio = AudioSegment.from_mp3(speechFilePath)
    except OSError as e:
        print "Could not process file".format(speechFilePath)
        raise e
    speech_audio = speech_audio.set_frame_rate(sample_rate)
    speech_array = np.array(speech_audio.get_array_of_samples())

    # normalize it
    max_speech = max(speech_array)
    speech_array = speech_array.astype('float32') / max_speech

    while True:
        try:
            # randomly grab a noisy file
            noise_ind = randint(0, len(noise_files) - 1)
            noiseFilePath = noise_files[noise_ind]
            [_, noiseFileName] = os.path.split(noiseFilePath)
            noise_audio = AudioSegment.from_wav(noiseFilePath)
            break
        except wave.Error as e:
            # some noise files can't be read as wav for strange reasons
            # keep trying other files
            pass

    noise_audio = noise_audio.set_frame_rate(sample_rate)
    noise_array = np.array(noise_audio.get_array_of_samples())
    # normalize it
    max_noise = max(noise_array)
    noise_array = noise_array.astype('float32') / max_noise

    if len(noise_array) < len(speech_array):
        # if noise is shorter than speech, just repeat it a few times. +1 just in case
        noise_array = np.tile(noise_array, int(ceil(len(speech_array) / float(len(noise_array)))))

    assert len(noise_array) >= len(speech_array)
    # if noise is longer than speech at this point, cut it short so they're the same length
    noise_array = noise_array[0:len(speech_array)]

    for noise_level in noise_levels:
        path_noisy_speech = '{}/{}_noisy{}/'.format(output_path, dataSet, noise_level)
        noise_level = noise_level / 100.0
        noisy_speech_array = (1 - noise_level) * speech_array + noise_level * noise_array
        noisy_speech_array *= max(max_noise, max_speech)  # scale things back up
        noisy_speech_array = noisy_speech_array.astype('int16')
        # save data
        combined = AudioSegment(noisy_speech_array.tobytes(2), sample_width=speech_audio.sample_width,
                                frame_rate=sample_rate, channels=speech_audio.channels)
        combined.export(path_noisy_speech + speechID + '.wav', format="wav")

    writeStr += '{},{}\n'.format(speechID, noiseFileName)

    if (i + 1) % 1000 == 0:
        flog.write(writeStr)
        writeStr = ''
        print "Finished {} with {} seconds".format(i + 1, time.time() - prev_time)
        prev_time = time.time()

flog.write(writeStr)
flog.close()
