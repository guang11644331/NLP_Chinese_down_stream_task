import tensorflow as tf
from sklearn import metrics
import tensorflow_addons as tfa


def _get_vitb_result(logits, ys, text_lens, model, evaluate=False):
    # 通过维特比算法计算一个batch的acc和preds和标签们（去掉pad的）
    acc_lines = 0
    if not evaluate:
        # 从每个batch里一行一行的读，相当于没有batch...
        for logit, y_line, text_len in zip(logits, ys, text_lens):
            viterbi_path, vitb_score = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
            correct_pred = tf.equal(viterbi_path, y_line[:text_len])
            acc_lines += tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

        acc_batch = acc_lines / len(ys)
        return viterbi_path, vitb_score, acc_batch
    else:
        pred_batch = []
        y_true_batch = []

        # 从每个batch里一行一行的读，相当于没有batch...
        for logit, y_line, text_len in zip(logits, ys, text_lens):
            viterbi_path, vitb_score = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
            correct_pred = tf.equal(viterbi_path, y_line[:text_len])
            acc_lines += tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
            pred_batch.extend(viterbi_path)
            y_true_batch.extend(y_line[:text_len])

        acc_batch = acc_lines / len(ys)
        return acc_batch, pred_batch, y_true_batch


def training(model, train_ds, val_ds, epochs, optimizer):
    for epoch in range(epochs):
        print(f'epoch[{epoch+1:02d}/{epochs:02d}]:')
        loss_all = []
        acc_all = []

        for i, batch in enumerate(train_ds):   # 每次取一个batch, shape (batch, 2, length)
            xs, ys = batch[:, 0], batch[:, 1]
            with tf.GradientTape() as tape:
                logits, log_likelihood, text_lens = model(xs, ys)
                loss = -1 * tf.reduce_mean(log_likelihood)
                loss_all.append(loss)
                viterbi_path, vitb_score, acc_batch = _get_vitb_result(logits, ys, text_lens, model)
                acc_all.append(acc_batch)

            # 分别对各个参数求导 d(loss) / d(w1)
            gradients = tape.gradient(loss, model.trainable_variables)
            # 梯度更新
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if i % 20 == 0:
                print(f"batch[{i:03d}]: loss={loss:.4f}, acc={acc_batch:.2%}")

        print(f"训练集: loss={tf.reduce_mean(loss_all):.4f}, acc = {tf.reduce_mean(acc_all):.2%}")
        evaluate(model, val_ds, data_name="验证集")


def evaluate(model, data, data_name="验证集", print_score=False, tag_names=None):
    loss_all = []
    acc_all = []

    if not print_score:
        for i, batch in enumerate(data):  # 每次取一个batch, shape (batch, 2, length)
            xs, ys = batch[:, 0], batch[:, 1]
            logits, log_likelihood, text_lens = model(xs, ys)
            loss = -1 * tf.reduce_mean(log_likelihood)
            loss_all.append(loss)
            viterbi_path, vitb_score, acc_batch = _get_vitb_result(logits, ys, text_lens, model)
            acc_all.append(acc_batch)

        print(f"{data_name}: loss={tf.reduce_mean(loss_all):.4f}, acc = {tf.reduce_mean(acc_all):.2%}")
    else:
        pred_all = []
        true_all = []
        for batch in data:
            xs, ys = batch[:, 0], batch[:, 1]
            logits, log_likelihood, text_lens = model(xs, ys)
            loss = -1 * tf.reduce_mean(log_likelihood)
            loss_all.append(loss)
            acc_batch, pred_batch, y_true_batch = _get_vitb_result(logits, ys, text_lens, model, evaluate=True)
            acc_all.append(acc_batch)
            pred_all.extend(pred_batch)
            true_all.extend(y_true_batch)

        print(f"{data_name}: loss={tf.reduce_mean(loss_all):.4f}, acc = {tf.reduce_mean(acc_all):.2%}")
        print(metrics.classification_report(true_all, pred_all, target_names=tag_names))
        print(metrics.confusion_matrix(true_all, pred_all))