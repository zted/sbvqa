import keras.backend as K


class TextMod(object):
    """ model initialization """

    def __init__(self, fc_dimension, vocab):
        self.fc_dimension = fc_dimension
        self.vocab = vocab

    """ build model """

    def build_model(self, max_len):
        from keras.models import Model
        from keras.layers import Embedding, Input, Dense, Dropout, LSTM, Lambda
        from keras.layers.merge import Multiply

        lstm_cells = 512
        fc_common_embedding_size = 512
        activation = 'tanh'
        dropout = 0.5
        emb_dim = 200
        vocab_size = self.vocab['q_vocab_size']
        output_classes = self.vocab['a_vocab_size']

        fc_input = Input(shape=(self.fc_dimension,), dtype='float32')
        fc_norm = Lambda(l2_norm, output_shape=(self.fc_dimension,))(fc_input)
        img_fc = Dense(fc_common_embedding_size, activation=activation, name='dense1_fc')(fc_norm)
        img_drop = Dropout(dropout)(img_fc)

        language_input = Input(shape=(max_len,), dtype='int32')
        l_in = Embedding(output_dim=emb_dim, input_dim=vocab_size + 1, input_length=max_len,
                         mask_zero='True', name='emb_fc')(language_input)
        lstm_fc = LSTM(lstm_cells, return_sequences=False, name='lstm_fc')(l_in)
        lstm_norm_fc = Lambda(l2_norm, output_shape=(lstm_cells,))(lstm_fc)
        lstm_drop_fc = Dropout(dropout)(lstm_norm_fc)
        v_q_fc = Dense(fc_common_embedding_size, activation=activation, name='dense2_fc')(lstm_drop_fc)

        fc_merged = Multiply([img_drop, v_q_fc])
        fc_merged_norm = Lambda(l2_norm, output_shape=(fc_common_embedding_size,))(fc_merged)
        fc_merged_dense = Dense(output_classes, activation=activation, name='dense_merge')(fc_merged_norm)
        fc_merged_drop = Dropout(dropout)(fc_merged_dense)
        fc_out = Dense(output_classes, activation='softmax', name='output_fc')(fc_merged_drop)

        model = Model(inputs=[language_input, fc_input], outputs=fc_out)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


class SpeechMod(object):
    """ model initialization """

    def __init__(self, img_dim):
        self.img_dim = img_dim

    """ build model """

    def build_model(self, output_classes):
        from keras.layers import BatchNormalization, Activation
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Lambda
        from keras.layers.convolutional import Conv1D
        from keras.layers.pooling import MaxPooling1D
        from keras.layers.recurrent import LSTM
        from keras.layers.merge import Multiply

        common_embedding_size = 512
        activation = 'tanh'
        dropout = 0.5

        image_model = Sequential()
        image_model.add(Lambda(l2_norm, input_shape=(self.img_dim,), output_shape=(self.img_dim,)))
        image_model.add(Dense(common_embedding_size, activation=activation))
        image_model.add(Dropout(dropout))

        speech_model = Sequential()
        speech_model.add(Conv1D(32, 64, strides=2, input_shape=(None, 1), name='conv_1'))
        speech_model.add(BatchNormalization())
        speech_model.add(Activation('relu'))
        speech_model.add(MaxPooling1D(pool_size=4))

        speech_model.add(Conv1D(64, 32, strides=2, name='conv_2'))
        speech_model.add(BatchNormalization())
        speech_model.add(Activation('relu'))
        speech_model.add(MaxPooling1D(pool_size=4))

        speech_model.add(Conv1D(128, 16, strides=2, name='conv_3'))
        speech_model.add(BatchNormalization())
        speech_model.add(Activation('relu'))
        speech_model.add(MaxPooling1D(pool_size=4))

        speech_model.add(Conv1D(256, 8, strides=2, name='conv_4'))
        speech_model.add(BatchNormalization())
        speech_model.add(Activation('relu'))
        speech_model.add(MaxPooling1D(pool_size=4))

        speech_model.add(Conv1D(512, 4, strides=2, name='last_conv'))
        speech_model.add(BatchNormalization())
        speech_model.add(Activation('relu'))

        speech_model.add(LSTM(common_embedding_size, return_sequences=False))
        speech_model.add(Lambda(l2_norm, output_shape=(common_embedding_size,)))
        speech_model.add(Dense(common_embedding_size, activation=activation))
        speech_model.add(Dropout(dropout))

        model = Sequential()
        model.add(Multiply([speech_model, image_model]))
        model.add(Lambda(l2_norm, output_shape=(common_embedding_size,)))
        model.add(Dense(common_embedding_size, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(output_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


def l2_norm(x):
    epsilon = 1e-4
    x_normed = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
    x = x / (x_normed + epsilon)
    return x
