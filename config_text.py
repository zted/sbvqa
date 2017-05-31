options = {}

# training parameters
options['max_epochs'] = 100  # 100,300
options['patience'] = 3
options['epochs_to_validate'] = 2
options['batch_size'] = 100
options['shuffle'] = True

# file paths
options['qah5'] = 'data_prepro.h5'
options['vocab_img_datafile'] = 'data_prepro.json'
options['img_file'] = 'data_img.h5'
options['test_questionfile'] = 'OpenEnded_mscoco_val2014_questions.json'
options['test_answerfile'] = 'mscoco_val2014_annotations.json'

options['dataset_root'] = './data/' # change this to the directory where you store all the files above
