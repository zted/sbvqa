import json
import sys

import h5py
import numpy as np
from keras.utils.np_utils import to_categorical

import utils as U
from config_text import *
from models import TextMod

try:
    loadWeightsFile = sys.argv[1]
except IndexError as e:
    print("Must provide as argument path to file containing model weights")
    raise e

dataset_root = options['dataset_root']
qah5_path = dataset_root + options['qah5']
img_path = dataset_root + options['img_file']

# -------------For evaluation on validation data--------------------
annFile = dataset_root + options['test_annfile']
quesFile = dataset_root + options['test_questionfile']

# ------------------Vocabulary indices--------------------------------
vocab_img_data_path = dataset_root + options['vocab_img_datafile']

# ------------------Loading image positions and question IDs-----------------
img_train_pos, img_val_pos, q_test_id = U.load_positions_ids(qah5_path)
q_train, q_val, a_train = U.load_questions_answers(qah5_path)
a_train = to_categorical(a_train - 1, a_train.max())
# the - 1 is for offset. so class #1 would be 0, it must be done because to_categorical starts at 0
q_maxlen = len(q_train[0])

# Allocate files that we want to save our weights, predictions, and log to
evaldump = './predictions/eval_predictions'
evaldump = U.determine_filename(evaldump, '.json')
f = open(evaldump, 'w')
f.close()

print("Dump predictions to {}"
      .format(evaldump))

# prepare data
print('Reading %s' % (vocab_img_data_path,))
vocab_data = json.load(open(vocab_img_data_path, 'r'))
with h5py.File(img_path, 'r') as hf:
    img_train = hf.get(u'images_train').value
    img_val = hf.get(u'images_test').value
img_feature_size = len(img_val[0])

print("Shapes:\nImage Train - {}\nImage Val - {}\nText Train - {}\nText Val - {}\nAns Train - {}"
      .format(img_train.shape, img_val.shape, q_train.shape, q_val.shape, a_train.shape))

vocab = {}
vocab['ix_to_word'] = vocab_data['ix_to_word']
vocab['q_vocab_size'] = len(vocab['ix_to_word'])
vocab['ix_to_ans'] = vocab_data['ix_to_ans']
vocab['a_vocab_size'] = len(vocab['ix_to_ans'])

# --------------------Training Parameters--------------------
batch_size = options.get('batch_size', 100)

print('Building model...')
mod = TextMod(img_feature_size, vocab)
model = mod.build_model(q_maxlen)

if loadWeightsFile is not None:
    model.load_weights(loadWeightsFile, by_name=True)
    print('Successfully loaded weights from {}'.format(loadWeightsFile))

val_array = np.arange(len(q_val))
nb_batch_val = int(np.ceil(len(val_array) / float(batch_size)))
print('Predicting...')
pred = np.array([], dtype='int')
for batch_index in range(0, nb_batch_val):
    batch_start = batch_index * batch_size
    batch_end = min(len(val_array), (batch_index + 1) * batch_size)
    current_batch_size = batch_end - batch_start
    val_indices = val_array[batch_start:batch_end]
    q_val_batch = np.array([q_val[i] for i in val_indices])
    i_pos_batch = [img_val_pos[i] for i in val_indices]
    i_val_batch = np.array([img_val[i - 1] for i in i_pos_batch])
    X_batch = [q_val_batch, i_val_batch]
    raw_pred = model.predict(X_batch, batch_size=current_batch_size, verbose=False)
    newpred = raw_pred.argmax(axis=-1)
    pred = np.concatenate((pred, newpred))
vqaEval = U.evaluate_and_dump_predictions(pred, q_test_id, quesFile, annFile, vocab['ix_to_ans'], evaldump)
val_acc = vqaEval.accuracy['overall']
print("Validation Accuracy Overall: {:.3f}\n".format(val_acc))
print("Accuracy Breakdown: {}\n".format(vqaEval.accuracy['perAnswerType']))
