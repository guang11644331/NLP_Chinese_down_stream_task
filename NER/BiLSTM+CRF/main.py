import utils
from Model import LSTM_CRF
import run
import tensorflow as tf


batch_size = 128
embed_dim = 100
maxlen = 200
rnn_hiden_size = 128
epochs = 3


if __name__ == '__main__':
    # load_data
    train = utils.load_data('data/china-people-daily-ner-corpus/example.train')
    dev = utils.load_data('data/china-people-daily-ner-corpus/example.dev')
    test = utils.load_data('data/china-people-daily-ner-corpus/example.test')

    token2idx, idx2token, tag2idx, idx2tag, real_maxlen = utils.data_preprocessing(train)  # len(token2idx) = 4314
    print(f'real_maxlen = {real_maxlen}')

    # padding
    train = utils.padding(train, token2idx, tag2idx, maxlen)
    dev = utils.padding(dev, token2idx, tag2idx, maxlen)
    test = utils.padding(test, token2idx, tag2idx, maxlen)

    train = utils._to_tensor(train, tf.int32)
    dev = utils._to_tensor(dev, tf.int32)
    test = utils._to_tensor(test, tf.int32)

    # to batch
    train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(10000).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices(dev).shuffle(2000).batch(batch_size * 2)
    test_ds = tf.data.Dataset.from_tensor_slices(test).shuffle(2000).batch(batch_size * 2)

    embedding_pretrained = utils.load_word2vec('data/embeddings/wiki_100.utf8', token2idx, embed_dim, 'data/embeddings/embed_mat.npy')

    model = LSTM_CRF(len(token2idx), embed_dim, maxlen, len(tag2idx), rnn_hiden_size, embedding_pretrained)
    optimizer = tf.keras.optimizers.Adam(lr=0.003)

    run.training(model, train_ds, dev_ds, epochs, optimizer)
    run.evaluate(model, test_ds, data_name="测试集")
    # # # save model
    # # print("\nsave model...")
    # # model.save_weights('model saved/')
    #
    # # load model
    # print("load model...")
    # model.load_weights('model saved/')
    # model.summary()
    run.evaluate(model, test_ds, data_name="测试集", print_score=True, tag_names=list(tag2idx.keys()))





    print("___" * 30)