from capsule import Capsule
from keras.layers import (GRU, LSTM, Activation, Bidirectional, Concatenate,
                          CuDNNGRU, CuDNNLSTM, Dense, Dropout, Embedding,
                          Flatten, Input, Lambda, Merge, Reshape,
                          SpatialDropout1D, TimeDistributed, concatenate)
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import (AveragePooling1D, Conv1D, Conv2D,
                                        MaxPooling1D, MaxPooling2D,
                                        ZeroPadding1D)
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras_attention import Attention, AttLayer
from keras_attention_context import AttentionWithContext
from keras_attention_deepmoji import AttentionWeightedAverage


def GRUCapsule(params):

    embed_dropout_rate = 0.1
    routings = 5
    num_capsule = 10
    dim_capsule = 16

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = SpatialDropout1D(embed_dropout_rate, name='embed_drop')(embed_input_)

    x = Bidirectional(
        CuDNNGRU(params['lstm_units'], return_sequences=True), name="bi_gru0")(x)
    capsule = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings,
                      share_weights=True)(x)
    x = Flatten()(capsule)

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def Conv2Dmodel(params):

    filter_sizes = [1, 2, 3, 5]
    num_filters = 32
    embed_dropout_rate = 0.1

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = SpatialDropout1D(embed_dropout_rate, name='embed_drop')(embed_input_)

    x = Reshape((params['sequence_length'], params['embedding_dim'], 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], params['embedding_dim'],), kernel_initializer='normal',
                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], params['embedding_dim'],), kernel_initializer='normal',
                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], params['embedding_dim'],), kernel_initializer='normal',
                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], params['embedding_dim'],), kernel_initializer='normal',
                    activation='elu')(x)

    maxpool_0 = MaxPooling2D(pool_size=(
        params['sequence_length'] - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(
        params['sequence_length'] - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(
        params['sequence_length'] - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPooling2D(pool_size=(
        params['sequence_length'] - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_, outputs=outp)
    model.compile(loss=params['loss'],
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def LSTMDeepmoji(params):

    return_attention = False
    embed_dropout_rate = 0.1

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(embed_dropout_rate, name='embed_drop')(x)

    lstm_0_output = Bidirectional(
        CuDNNLSTM(params['lstm_units'], return_sequences=True), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(CuDNNLSTM(
        params['lstm_units'], return_sequences=True), name="bi_lstm_1")(lstm_0_output)

    x = concatenate([lstm_1_output, lstm_0_output, x])
    x = AttentionWeightedAverage(
        name='attlayer', return_attention=return_attention)(x)
    if return_attention:
        x, weights = x

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


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
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
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
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMHierarchical(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    l_lstm = Bidirectional(
        CuDNNLSTM(params['lstm_units'], return_sequences=True))(embed_input_)
    l_dense = TimeDistributed(Dense(256))(l_lstm)
    l_att = GlobalMaxPooling1D()(l_dense)
    sentEncoder = Model(input_, l_att)

    review_input = Input(shape=(1, params['sequence_length']))
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(
        CuDNNLSTM(params['lstm_units'], return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(256))(l_lstm_sent)
    l_att = GlobalMaxPooling1D()(l_dense_sent)
    x = Dense(6, activation='sigmoid')(l_att)

    model = Model(inputs=review_input, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUHierarchical(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    l_lstm = Bidirectional(
        CuDNNGRU(params['lstm_units'], return_sequences=True))(embed_input_)
    l_dense = TimeDistributed(Dense(256))(l_lstm)
    l_att = GlobalMaxPooling1D()(l_dense)
    sentEncoder = Model(input_, l_att)

    review_input = Input(shape=(1, params['sequence_length']))
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(
        CuDNNGRU(params['lstm_units'], return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(256))(l_lstm_sent)
    l_att = GlobalMaxPooling1D()(l_dense_sent)
    x = Dense(6, activation='sigmoid')(l_att)

    model = Model(inputs=review_input, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMmax(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUmax(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMconcat(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    merge_layer = concatenate([x1, x2])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUconcat(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    merge_layer = concatenate([x1, x2])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUconvconcat(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
        x = Conv1D(64, kernel_size=3, padding='valid',
                   kernel_initializer='glorot_uniform')(x)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
        x = Conv1D(64, kernel_size=3, padding='valid',
                   kernel_initializer='glorot_uniform')(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    merge_layer = concatenate([x1, x2])

    # x = Dropout(params['dropout_rate'])(merge_layer)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMconcat2(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = AttentionWithContext()(x)
    merge_layer = concatenate([x1, x2, x3])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    return model


def GRUconcat2(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = AttentionWithContext()(x)
    merge_layer = concatenate([x1, x2, x3])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    return model


def GRUconvconcat2(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
        x = Conv1D(64, kernel_size=3, padding='valid',
                   kernel_initializer='glorot_uniform')(x)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
        x = Conv1D(64, kernel_size=3, padding='valid',
                   kernel_initializer='glorot_uniform')(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = AttentionWithContext()(x)
    merge_layer = concatenate([x1, x2, x3])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    return model


def GRUconcat3(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = AttentionWithContext()(x)
    x4 = Attention(params['sequence_length'])(x)
    merge_layer = concatenate([x1, x2, x3, x4])

    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
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
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x = AttentionWithContext()(x)
    # x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUattention(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(embed_input_)
    x = AttentionWithContext()(x)
    # x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUdeep(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(0.1, name='embed_drop')(x)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(x)
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
    else:
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(x)
        x = CuDNNGRU(params['lstm_units'],
                     return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUConvdeep(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(0.1, name='embed_drop')(x)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(x)
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(x)
        x1 = GlobalMaxPooling1D()(x)
        x2 = GlobalAveragePooling1D()(x)

        x_conv = Conv1D(64, kernel_size=3, padding='valid',
                        kernel_initializer='glorot_uniform')(x)
        x1_conv = GlobalMaxPooling1D()(x_conv)
        x2_conv = GlobalAveragePooling1D()(x_conv)

    merge_layer = concatenate([x1, x2, x1_conv, x2_conv])
    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def GRUConvdeep3(params):

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(0.1, name='embed_drop')(x)

    if params['bidirectional']:
        x_g1 = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(x)
        x_l1 = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(x_g1)
        x_g2 = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(x_l1)
        x1 = GlobalMaxPooling1D()(x_g2)
        x2 = GlobalAveragePooling1D()(x_g2)
        merge_layer = concatenate([x_g1, x_l1, x_g2])

        x_conv = Conv1D(64, kernel_size=3, padding='valid',
                        kernel_initializer='glorot_uniform')(merge_layer)
        x1_conv = GlobalMaxPooling1D()(x_conv)
        x2_conv = GlobalAveragePooling1D()(x_conv)

    merge_layer2 = concatenate([x1_conv, x2_conv])
    x = Dropout(params['dropout_rate'])(merge_layer2)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMdeep(params):

    embed_dropout_rate = 0.1

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(embed_dropout_rate, name='embed_drop')(x)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(x)
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(x)
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def LSTMdeep2(params):

    embed_dropout_rate = 0.1

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)
    x = Activation('tanh')(embed_input_)
    x = SpatialDropout1D(embed_dropout_rate, name='embed_drop')(x)

    if params['bidirectional']:
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True,
                      kernel_initializer='he_uniform'))(x)
        x = Bidirectional(
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True,
                      kernel_initializer='he_uniform'))(x)
        x1 = GlobalMaxPooling1D()(x)
        x2 = GlobalAveragePooling1D()(x)
        x3 = AttentionWithContext()(x)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True,
                      kernel_initializer='he_uniform')(x)
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True,
                      kernel_initializer='he_uniform')(x)
        x = GlobalMaxPooling1D()(x)

    merge_layer = concatenate([x1, x2, x3])
    x = Dropout(params['dropout_rate'])(merge_layer)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
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
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
    x = Attention(params['sequence_length'])(x)
    # x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'],
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
            CuDNNLSTM(params['lstm_units'],
                      return_sequences=True))(embed_input_)
    else:
        x = CuDNNLSTM(params['lstm_units'],
                      return_sequences=True)(embed_input_)
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
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=[input_, mlp_input], outputs=x)
    model.compile(loss=params['loss'],
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
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def ConvKim(params):

    convs = []
    filter_sizes = [3, 4, 5]

    Embedding_layer = Embedding(params['nb_words'],
                                params['embedding_dim'],
                                weights=[params['embedding_matrix']],
                                input_length=params['sequence_length'],
                                trainable=False)

    input_ = Input(shape=(params['sequence_length'], ))
    embed_input_ = Embedding_layer(input_)

    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz,
                        activation='relu')(embed_input_)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(10)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(6, activation='sigmoid')(l_dense)

    model = Model(inputs=input_, outputs=preds)
    model.compile(loss=params['loss'],
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
    glob1a = GlobalMaxPooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalMaxPooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalMaxPooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalMaxPooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalMaxPooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalMaxPooling1D()(conv6a)

    merge_layer = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])

    x = Dropout(0.2)(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'], optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


def Conv1DGRUbranched(params):

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
    glob1a = GlobalMaxPooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalMaxPooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalMaxPooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalMaxPooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalMaxPooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalMaxPooling1D()(conv6a)

    if params['bidirectional']:
        rnn_branch = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        rnn_branch = CuDNNGRU(params['lstm_units'],
                              return_sequences=True)(embed_input_)
    rnn_branch1 = AttentionWithContext()(rnn_branch)
    rnn_branch2 = GlobalMaxPooling1D()(rnn_branch)

    merge_layer = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a,
                               rnn_branch1, rnn_branch2])

    x = Dropout(0.2)(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=params['loss'], optimizer=params['optimizer'],
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
    glob1a = GlobalMaxPooling1D()(conv1a)

    conv2a = conv2(embed_input_)
    glob2a = GlobalMaxPooling1D()(conv2a)

    conv3a = conv3(embed_input_)
    glob3a = GlobalMaxPooling1D()(conv3a)

    conv4a = conv4(embed_input_)
    glob4a = GlobalMaxPooling1D()(conv4a)

    conv5a = conv5(embed_input_)
    glob5a = GlobalMaxPooling1D()(conv5a)

    conv6a = conv6(embed_input_)
    glob6a = GlobalMaxPooling1D()(conv6a)

    if params['bidirectional']:
        rnn_branch = Bidirectional(
            CuDNNGRU(params['lstm_units'],
                     return_sequences=True))(embed_input_)
    else:
        rnn_branch = CuDNNGRU(params['lstm_units'],
                              return_sequences=True)(embed_input_)
    # rnn_branch = AttentionWithContext()(rnn_branch)
    rnn_branch = GlobalMaxPooling1D()(rnn_branch)

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
    model.compile(loss=params['loss'], optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model
