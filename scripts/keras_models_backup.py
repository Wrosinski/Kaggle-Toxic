from keras.layers import (GRU, LSTM, Bidirectional, CuDNNGRU, CuDNNLSTM, Dense,
                          Dropout, Embedding, Flatten, Input, Lambda, Reshape,
                          concatenate)
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import (AveragePooling1D, Conv1D, MaxPooling1D,
                                        ZeroPadding1D)
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras_attention import Attention
from keras_attention_context import AttentionWithContext


def LSTMbasic(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            LSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        x = LSTM(params['lstm_units'], return_sequences=True)(embed_input_)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMattention(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            LSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        x = LSTM(params['lstm_units'], return_sequences=True)(embed_input_)
    x = AttentionWithContext()(x)
    # x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMattentionV2(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            LSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        x = LSTM(params['lstm_units'], return_sequences=True)(embed_input_)
    x = Attention(params['sequence_length'])(x)
    # x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUbasic(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            GRU(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        x = GRU(params['lstm_units'], return_sequences=True)(embed_input_)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def MLPbasic(params):

    input_ = Input(shape=(params['num_columns'],), sparse=False)

    x = BatchNormalization()(input_)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMattentionBranched(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            LSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        x = LSTM(params['lstm_units'], return_sequences=True)(embed_input_)
    x = AttentionWithContext()(x)

    mlp_input = Input(shape=(params['num_columns'],))
    mlp = BatchNormalization()(mlp_input)
    mlp = Dense(256)(mlp)
    mlp = PReLU()(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dense(256)(mlp)
    mlp = PReLU()(mlp)

    merge_layer = concatenate([x, mlp])
    x = Dropout(0.1)(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=[input_, mlp_input], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def Conv1DQuora(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4,
                   padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6,
                   padding='same', activation='relu')

    conv1a = conv1(embed_input_)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalAveragePooling1D()(conv6a)

    merge_layer = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])

    x = Dropout(0.2)(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def Conv1DLSTMbranched(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4,
                   padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6,
                   padding='same', activation='relu')

    conv1a = conv1(embed_input_)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalAveragePooling1D()(conv6a)

    if params['bidirectional']:
        rnn_branch = Bidirectional(
            LSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        rnn_branch = LSTM(params['lstm_units'],
                          return_sequences=True)(embed_input_)
    rnn_branch = AttentionWithContext()(rnn_branch)
    # rnn_branch = GlobalAveragePooling1D()(rnn_branch)

    merge_layer = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a,
                               rnn_branch])

    x = Dropout(0.2)(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def Conv1DLSTMbranchedV2(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4,
                   padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6,
                   padding='same', activation='relu')

    conv1a = conv1(embed_input_)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalAveragePooling1D()(conv6a)

    if params['bidirectional']:
        rnn_branch = Bidirectional(
            GRU(params['lstm_units'], return_sequences=True))(embed_input_)
    else:
        rnn_branch = GRU(params['lstm_units'],
                         return_sequences=True)(embed_input_)
    # rnn_branch = AttentionWithContext()(rnn_branch)
    rnn_branch = GlobalAveragePooling1D()(rnn_branch)

    mlp_input = Input(shape=(params['num_columns'],))
    mlp = BatchNormalization()(mlp_input)
    mlp = Dense(256)(mlp)
    mlp = PReLU()(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dense(256)(mlp)
    mlp = PReLU()(mlp)

    merge_layer = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a,
                               rnn_branch, mlp])

    x = Dropout(0.2)(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=[input_, mlp_input], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model
