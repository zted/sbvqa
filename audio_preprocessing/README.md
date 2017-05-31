Contains files needed to preprocess the speech data, or to preprocess data transcribed from the speech files.

After process_transcribed_text.py is used, the resulting file can be used in TextMod. You will have to write a few lines to read the resulting hdf5 file, such as q_val = hf.File('transcribed_questions').get('ques_test') or something along those lines.