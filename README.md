# Speech-Based Visual Question Answering. Releasing public code for the [paper](https://arxiv.org/abs/1705.00464).

### Preprocessing
follow the standard procedure preprocessing used [here](https://github.com/VT-vision-lab/VQA_LSTM_CNN).

put all the files in ./data

### To train a model
edit config_text.py or config_speech.py so that the paths are pointing in the correct directory. run
```
python train_TextMod.py
```
the same call is used for SpeechMod

### To just evaluate, or to make predictions on test-dev or test
edit config_text.py or config_speech.py so that it uses the correct set of images, vocab, questions and answers.
notice that when evaluating on test-dev or test, answers are not provided so you have to evaluate on the VQA server.
in this case just set the answers file path to the validation file. to run evaluation code, run
```
python eval_TextMod.py path_to_weights.h5
```
pre-trained weights must be provided as an argument. if evaluating on validation set, the program will terminate
successfully and display the results. if evaluating on test-dev or test, the program will terminate unsuccessfully
because the answers do not match the questions. nevertheless, the predictions will have been dumped in 
./predictions/... upload this file to the VQA server for evaluation

### To download the generated spoken questions from Amazon Polly
Go to this link: http://data.vision.ee.ethz.ch/daid/VQA/SpeechVQA.zip

The questions are named "question_id.mp3"