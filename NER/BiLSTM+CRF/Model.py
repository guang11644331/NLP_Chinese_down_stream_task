import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import tensorflow_addons as tfa


class LSTM_CRF(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, maxlen, n_tags, rnn_hiden_size, embedding_pretrained):
        super(LSTM_CRF, self).__init__()
        self.embed = Embedding(vocab_size, embed_dim, input_length=maxlen,
                               weights=[embedding_pretrained], trainable=True)
        self.bilstm = Bidirectional(LSTM(rnn_hiden_size, return_sequences=True), merge_mode='concat')
        self.fc = Dense(n_tags)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(n_tags, n_tags)),
                                             trainable=False)
        # self.dropout = Dropout(0.3)

    def call(self, inputs, labels, training=None, mask=None):
        x = tf.math.not_equal(inputs, 0)
        x = tf.cast(x, dtype=tf.int32)
        text_lens = tf.math.reduce_sum(x, axis=-1)  # 计算每个句子去掉pad的长度

        embeddings = self.embed(inputs)  # inputs shape = (batch, length, embed_dim)
        hidden_states = self.bilstm(embeddings)
        # hidden_states = self.dropout(hidden_states, training)
        logits = self.fc(hidden_states)
        log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits, labels, text_lens)

        return logits, log_likelihood, text_lens