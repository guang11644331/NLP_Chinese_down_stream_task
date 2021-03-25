import pandas as pd
import numpy as np
import jieba.posseg as posg
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, concatenate
import tensorflow as tf
from tensorflow.keras import optimizers


def cut_word(text):
    words = posg.cut(text)
    # 去标点, 数字, 去 '是， 的, 了...'
    w = []
    stopwords = ['是', '的', '了']
    stopflags = ['x', 'm']
    for word, flag in words:
        if (word not in stopwords) and (flag not in stopflags):
            w.append(word)
    return ' '.join(w)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed = Embedding(tokenizer.num_words+1, 50)
        self.lstm = LSTM(64)
        self.dense = Dense(3, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        input_1 = inputs[0]
        input_2 = inputs[1]
        embedding_1 = self.embed(input_1)
        embedding_2 = self.embed(input_2)
        output_1 = self.lstm(embedding_1)
        output_2 = self.lstm(embedding_2)
        merged = concatenate([output_1, output_2])  # 合并
        outputs = self.dense(merged)

        return outputs


def training(model, train_ds, val_ds, epochs, optimizer, loss_func, metrics):
    for epoch in range(epochs):
        print(f'[{epoch+1:02d}/{epochs:02d}]:')

        total_loss = []
        metrics.reset_states()

        for i, (xs1, xs2, ys) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                pred = model([xs1, xs2])  # shape [batch, 10]
                loss = loss_func(ys, pred)
                total_loss.append(loss)
                metrics(ys, pred)

            # 分别对各个参数求导 d(loss) / d(w1)
            grads = tape.gradient(loss, model.trainable_variables)
            # 梯度更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # if i % 100 == 0:
            #     print(f"[{i:03d}]: loss={loss:.4f}")

        print(f"train_loss={tf.reduce_mean(total_loss):.4f}, train_acc = {metrics.result():.2%}")
        print("验证集:")
        testing(model, val_ds, loss_func, metrics)


def testing(model, data, loss_func, metrics):
    total_loss = 0
    metrics.reset_states()

    for xs1, xs2, ys in data:
        pred = model([xs1, xs2])
        loss = loss_func(ys, pred)
        total_loss += loss
        metrics(ys, pred)

    print(f"loss={tf.reduce_mean(total_loss):.4f}, acc = {metrics.result():.2%}")


batch_size = 128


if __name__ == '__main__':
    df = pd.read_csv("data/fake-news-pair-classification-challenge/train.csv", nrows=40000)
    train = df.loc[:, ['title1_zh', 'title2_zh', 'label']]

    # 1. 分词，去停用词，去标点
    train['title1_tokenized'] = train['title1_zh'].apply(cut_word)
    train['title2_tokenized'] = train['title2_zh'].apply(cut_word)

    # 2. 构建词典，padding
    x = pd.concat([train['title1_tokenized'], train['title2_tokenized']])
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x)
    encoded1 = tokenizer.texts_to_sequences(train['title1_tokenized'])
    encoded2 = tokenizer.texts_to_sequences(train['title2_tokenized'])
    input_len = 25
    pad1 = pad_sequences(encoded1, maxlen=input_len)
    pad2 = pad_sequences(encoded2, maxlen=input_len)
    label = {'unrelated': 0, 'agreed': 1, 'disagreed': 2}
    y = train['label'].apply(lambda x: label[x])

    x1_train_all, x1_test, x2_train_all, x2_test, y_train_all, y_test = train_test_split(pad1, pad2, y)
    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train_all, x2_train_all, y_train_all)

    x1_train = tf.convert_to_tensor(x1_train, dtype=tf.float32)
    x2_train = tf.convert_to_tensor(x2_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    x1_val = tf.convert_to_tensor(x1_val, dtype=tf.float32)
    x2_val = tf.convert_to_tensor(x2_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

    x1_test = tf.convert_to_tensor(x1_test, dtype=tf.float32)
    x2_test = tf.convert_to_tensor(x2_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((x1_train, x2_train, y_train)).shuffle(5000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x1_val, x2_val, y_val)).shuffle(5000).batch(batch_size * 2)
    test_ds = tf.data.Dataset.from_tensor_slices((x1_test, x2_test, y_test)).shuffle(5000).batch(batch_size * 2)

    model = MyModel()

    optimizer = optimizers.Adam(lr=1e-3)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()  # 自动叠加并且平均

    training(model, train_ds, val_ds, 10, optimizer, loss_func, metrics)
    print("测试集：")
    testing(model, test_ds, loss_func, metrics)












    print("___" * 30)