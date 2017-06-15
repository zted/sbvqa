import json
import sys
import time

import h5py
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import utils as U
from config_speech import *
from models import SpeechMod

try:
    loadWeightsFile = sys.argv[1]
except IndexError as e:
    print("Must provide as argument path to file containing model weights")
    raise e

dataset_root = options['dataset_root']
qah5_path = dataset_root + options['qah5']
img_path = dataset_root + options['img_file']

# -------------For evaluation on validation data--------------------
ansFile = dataset_root + options['test_answerfile']
quesFile = dataset_root + options['test_questionfile']
speech_train_file = dataset_root + options['wav_train']
speech_val_file = dataset_root + options['wav_test']
# ------------------Vocabulary indices--------------------------------
vocab_img_data_path = dataset_root + options['vocab_img_datafile']

# ------------------Loading image positions and question IDs-----------------
img_train_pos, img_val_pos, q_test_id = U.load_positions_ids(qah5_path)
_, _, a_train = U.load_questions_answers(qah5_path)
a_train = to_categorical(a_train - 1, a_train.max())
# the - 1 is for offset. so class #1 would be 0, it must be done because to_categorical starts at 0


# Allocate file that we dump predictions to
evaldump = './predictions/eval_predictions'
evaldump = U.determine_filename(evaldump, '.json')
f = open(evaldump, 'w')
f.close()

speech_train = h5py.File(speech_train_file, 'r').get('train')
speech_val = h5py.File(speech_val_file, 'r').get('val')

# prepare data
print('Reading %s' % (vocab_img_data_path,))
vocab_data = json.load(open(vocab_img_data_path, 'r'))
with h5py.File(img_path, 'r') as hf:
    img_train = hf.get(u'images_train').value
    img_val = hf.get(u'images_test').value
img_feature_size = len(img_val[0])

print("Shapes:\nImage Train - {}\nImage Val - {}\nSpeech Train - {}\nSpeech Val - {}\nAns Train - {}"
      .format(img_train.shape, img_val.shape, speech_train.shape, speech_val.shape, a_train.shape))

vocab = {}
vocab['ix_to_word'] = vocab_data['ix_to_word']
vocab['q_vocab_size'] = len(vocab['ix_to_word'])
vocab['ix_to_ans'] = vocab_data['ix_to_ans']
vocab['a_vocab_size'] = len(vocab['ix_to_ans'])

print('Building model...')
mod = SpeechMod(img_feature_size)
model = mod.build_model(len(a_train[0]))

model.load_weights(loadWeightsFile)
print('Successfully loaded weights from {}'.format(loadWeightsFile))

# --------------------Training Parameters--------------------
batch_size = options.get('batch_size', 100)

val_array = np.arange(len(speech_val))
total_time = time.time()
nb_batch_val = int(np.ceil(len(val_array) / float(batch_size)))

print('Predicting...')
pred = np.array([], dtype='int')
for batch_index in range(0, nb_batch_val):
    batch_start = batch_index * batch_size
    batch_end = min(len(val_array), (batch_index + 1) * batch_size)
    current_batch_size = batch_end - batch_start
    val_indices = val_array[batch_start:batch_end]
    q_val_batch = np.array([speech_val[i] for i in val_indices])
    i_pos_batch = [img_val_pos[i] for i in val_indices]
    i_val_batch = np.array([img_val[i - 1] for i in i_pos_batch])
    # i - 1 because positions were recorded as starting from 1
    q_val_batch = pad_sequences(q_val_batch, maxlen=None, padding='post', value=0.0, dtype='float32')
    q_val_batch = np.array([256.0 * s / s.max() for s in q_val_batch])
    q_val_batch = np.reshape(q_val_batch, (q_val_batch.shape[0], q_val_batch.shape[1], 1))
    X_batch = [q_val_batch, i_val_batch]
    raw_pred = model.predict(X_batch, batch_size=current_batch_size, verbose=False)
    newpred = raw_pred.argmax(axis=-1)
    pred = np.concatenate((pred, newpred))
vqaEval = U.evaluate_and_dump_predictions(pred, q_test_id, quesFile, ansFile, vocab['ix_to_ans'], evaldump)
val_acc = vqaEval.accuracy['overall']

print("Validation Accuracy Overall: {:.3f}\n".format(val_acc))
print("Accuracy Breakdown: {}\n".format(vqaEval.accuracy['perAnswerType']))
