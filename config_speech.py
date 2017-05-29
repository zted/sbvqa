import socket
options = {}

# training parameters
options['max_epochs']=300 # 100,300
options['patience']= 3# means n*period epochs
options['epochs_to_validate'] = 2
options['batch_size']=100
options['shuffle']=True

# file paths
options['qah5'] = 'data_LSTMCNN_trainval.h5'
options['vocab_img_datafile'] = 'data_LSTMCNN_trainval.json'
options['img_train'] = 'img_fc_trainval.h5'
options['wav_train'] = 'waveform_train.h5'
options['wav_test'] = 'waveform_val.h5'
options['test_annfile'] = 'mscoco_val2014_annotations.json'
options['test_questionfile'] = 'OpenEnded_mscoco_val2014_questions.json'
options['validate'] = True

hostname = socket.gethostname()
if hostname == 'crunchy':
    options['dataset_root'] ='/home/ted/research/data/'
elif hostname == 'bilbo':
    options['dataset_root'] = '/export/home1/NoCsBack/hci/ted/data/'
else:
    options['dataset_root'] = '/esat/tiger/r0602652/vqadata/'
    options['progress_bar'] = False
    options['log_output'] = True