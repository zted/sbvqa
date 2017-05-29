import json
import os
import logging

import h5py

from evalTools.vqaEval import VQAEval
from evalTools.vqa import VQA


def load_questions_answers(filePath):
    with h5py.File(filePath, 'r') as hf:
        questions_train = hf.get(u'ques_train').value
        questions_test = hf.get(u'ques_test').value
        ans_train = hf.get(u'answers').value
    return questions_train, questions_test, ans_train


def load_positions_ids(filePath):
    with h5py.File(filePath, 'r') as hf:
        img_train_pos = hf.get(u'img_pos_train').value
        img_val_pos = hf.get(u'img_pos_test').value
        q_test_id = hf.get(u'question_id_test').value
    return img_train_pos, img_val_pos, q_test_id


def evaluate_and_dump_predictions(pred, qids, qfile, afile, ix_ans_dict, filename):
    """
    dumps predictions to some default file
    :param pred: list of predictions, like [1, 2, 3, 2, ...]. one number for each example
    :param qids: question ids in the same order of predictions, they need to align and match
    :param qfile:
    :param afile:
    :param ix_ans_dict:
    :return:
    """
    # assert len(pred) == len(qids), "Number of predictions need to match number of question IDs"
    answers = []
    for i, val in enumerate(pred):
        qa_pair = {}
        qa_pair['question_id'] = int(qids[i])
        qa_pair['answer'] = ix_ans_dict[str(val + 1)]  # note indexing diff between python and torch
        answers.append(qa_pair)
    fod = open(filename, 'wb')
    json.dump(answers, fod)
    fod.close()
    # VQA evaluation
    vqa = VQA(afile, qfile)
    vqaRes = vqa.loadRes(filename, qfile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    return vqaEval


def determine_filename(filename, extension=''):
    keep_iterating = True
    count = 0
    while keep_iterating:
        # making sure to not save the weights as the same as an existing one
        count += 1
        unused_name = filename + str(count) + extension
        if not os.path.isfile(unused_name):
            keep_iterating = False
    return unused_name


def defineOutputFiles():
    import __main__ as main

    # Find the file names that we want to save weights and evaluation to. Make a file
    # there so that other programs that just begin to run won't use the same name
    tmpweights = './weights/weights'
    tmpweights = determine_filename(tmpweights, '.hdf5')
    f = open(tmpweights, 'w')
    f.close()

    evaldump = './predictions/predictions'
    evaldump = determine_filename(evaldump, '.json')
    f = open(evaldump, 'w')
    f.close()

    log_name = os.path.basename(main.__file__)
    log_output = './outputs/' + log_name
    log_output = determine_filename(log_output, '.log')
    f = open(log_output, 'w')
    f.close()

    return tmpweights, evaldump, log_output


def build_logger(logFile, loggerName='SomeLog'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logFile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Now, we can log to the root logger, or any other logger. First the root...
    logging.info('Logger was successfully setup.')
    logger = logging.getLogger(loggerName)
    # Now, define a couple of other loggers which might represent areas in your
    # application:
    return logger
