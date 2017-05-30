import json
import sys
import time

import h5py
import numpy as np
from keras.utils.generic_utils import Progbar
from keras.utils.np_utils import to_categorical

import utils as U
from config_text import *
from models import TextMod

try:
    # to start training from pre-existing model
    loadWeightsFile = sys.argv[1]
except IndexError as e:
    loadWeightsFile = None

dataset_root = options['dataset_root']
qah5_path = dataset_root + options['qah5']
img_path = dataset_root + options['img_file']

# -------------For evaluation on validation data--------------------
ansFile = dataset_root + options['test_answerfile']
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
saveNetWeights, evaldump, log_output = U.defineOutputFiles()
logger = U.build_logger(log_output, log_output)
logger.info("Save weights to {}\nDump predictions to {}"
            .format(saveNetWeights, evaldump))

# prepare data
logger.info('Reading %s' % (vocab_img_data_path,))
vocab_data = json.load(open(vocab_img_data_path, 'r'))
with h5py.File(img_path, 'r') as hf:
    img_train = hf.get(u'images_train').value
    img_val = hf.get(u'images_test').value
img_feature_size = len(img_val[0])

logger.info("Shapes:\nImage Train - {}\nImage Val - {}\nText Train - {}\nText Val - {}\nAns Train - {}"
            .format(img_train.shape, img_val.shape, q_train.shape, q_val.shape, a_train.shape))

vocab = {}
vocab['ix_to_word'] = vocab_data['ix_to_word']
vocab['q_vocab_size'] = len(vocab['ix_to_word'])
vocab['ix_to_ans'] = vocab_data['ix_to_ans']
vocab['a_vocab_size'] = len(vocab['ix_to_ans'])

# --------------------Training Parameters--------------------
batch_size = options.get('batch_size', 100)
nb_epoch = options.get('max_epochs', 100)
shuffle = options.get('shuffle', True)
max_patience = options.get('patience', 5)

logger.info('Building model...')
mod = TextMod(img_feature_size, vocab)
model = mod.build_model(q_maxlen)

if loadWeightsFile is not None:
    model.load_weights(loadWeightsFile, by_name=True)
    logger.info('Successfully loaded weights from {}'.format(loadWeightsFile))

train_array = np.arange(len(q_train))
val_array = np.arange(len(q_val))
nb_batch_train = int(np.ceil(len(train_array) / float(batch_size)))
nb_batch_val = int(np.ceil(len(val_array) / float(batch_size)))
best_yet = 0
patience = 0
total_time = time.time()
logger.info('Train...')
for e in range(nb_epoch):
    logger.info("Training epoch {}".format(e + 1))
    pbar = Progbar(1 + len(q_train) / batch_size)
    if shuffle:
        np.random.shuffle(train_array)
    start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    # Need to manually divide up training data in batches because it's too big to load
    for batch_index in range(0, nb_batch_train):
        batch_start = batch_index * batch_size
        batch_end = min(len(train_array), (batch_index + 1) * batch_size)
        current_batch_size = batch_end - batch_start
        training_indices = train_array[batch_start:batch_end]
        q_train_batch = np.array([q_train[i] for i in training_indices])
        i_pos_batch = [img_train_pos[i] for i in training_indices]
        i_train_batch = np.array([img_train[i - 1] for i in i_pos_batch])
        # i - 1 because positions were recorded as starting from 1
        a_batch = np.array([a_train[i] for i in training_indices])
        X_batch = [q_train_batch, i_train_batch]
        history = model.fit(X_batch, a_batch, batch_size=current_batch_size, epochs=1,
                            verbose=False)
        train_acc += history.history['acc'][-1]
        train_loss += history.history['loss'][-1]
        # because we're manually doing batch training but still want accuracy,
        # we tally up the accuracies from each batch and average them later.
        pbar.update(batch_index)

    logger.info("\nFinished training epoch {}. Accuracy = {:.3f}. Loss = {:.3f}"
                .format(e + 1, train_acc / nb_batch_train, train_loss / nb_batch_train))
    # the accuracy should be slightly higher, because we are rounding up the number of examples when
    # we multiply batch size * nb_batch
    logger.info("Time taken to train epoch: {}".format(int(time.time() - start_time)))

    if (e + 1) % options['epochs_to_validate'] == 0:
        # we validate every few epochs
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
        vqaEval = U.evaluate_and_dump_predictions(pred, q_test_id, quesFile, ansFile, vocab['ix_to_ans'], evaldump)
        val_acc = vqaEval.accuracy['overall']
        logger.info("Validation Accuracy Overall: {:.3f}\n".format(val_acc))
        logger.info("Accuracy Breakdown: {}\n".format(vqaEval.accuracy['perAnswerType']))
        if val_acc > best_yet:
            logger.info(
                'Accuracy improved from {} to {}, saving weights to {}'.format(best_yet, val_acc, saveNetWeights))
            best_yet = val_acc
            model.save_weights(saveNetWeights, overwrite=True)
            patience = 0
        else:
            patience += 1
    if patience > max_patience:
        logger.info('Out of patience. No improvement after {} epochs'.format(patience * options['epochs_to_validate']))
        break

logger.info("Best accuracy achieved was {}".format(best_yet))
logger.info("Total time taken was {} seconds".format(time.time() - total_time))
