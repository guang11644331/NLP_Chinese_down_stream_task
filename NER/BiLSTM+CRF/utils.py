from tqdm import tqdm
import tensorflow as tf
import os
import numpy as np


# 加载并处理数据
def load_data(file_path):
    sentence = []
    tags = []
    data = []
    # i = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # i+=1
            # if i == 1000000:
            #     break
            line = line.strip()
            if len(line.split()) == 2:
                word, tag = line.split()
                sentence.append(word)
                tags.append(tag)
            # line 为空表示一行结束
            elif not line:
                if sentence:
                    data.append([sentence, tags])
                    sentence = []
                    tags = []
            else:
                print(f'未知情况...{line}')
    return data


def data_preprocessing(data):
    """
    :return: token2idx, idx2token, tag2idx, idx2tag, maxlen
    """
    maxlen = max([len(line[0]) for line in data])
    sent_flat = []
    tags_flat = []
    for line in data:
        sent_flat.extend(line[0])
        tags_flat.extend(line[1])
    unique_tokens = list(set(sent_flat))
    unique_tags = list(set(tags_flat))

    token2idx = {'[PAD]': 0, '[UNK]': 1}
    token2idx.update({token: i+2 for i, token in enumerate(unique_tokens)})
    idx2token = {i:token for token, i in token2idx.items()}

    tag2idx = {'O': 0}
    unique_tags.remove('O')
    tag2idx.update({tag: i+1 for i, tag in enumerate(unique_tags)})
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    return token2idx, idx2token, tag2idx, idx2tag, maxlen


def padding(data, token2idx, tag2idx, maxlen):
    """
    sentence 需要pad [UNK], tag 不需要
    """
    data_padded = []
    for line in data:
        seq_len = len(line[0])
        # sentence
        token_ids = [token2idx.get(token, token2idx['[UNK]']) for token in line[0]]
        # tags
        tag_ids = [tag2idx.get(tag) for tag in line[1]]

        # padding
        if seq_len < maxlen:
            token_ids.extend([token2idx['[PAD]']] * (maxlen - seq_len))
            tag_ids.extend([tag2idx['O']] * (maxlen - seq_len))
        else:
            token_ids = token_ids[:maxlen]
            tag_ids = tag_ids[:maxlen]

        data_padded.append([token_ids, tag_ids])
    return data_padded


def _to_tensor(mat, dtype):
    return tf.convert_to_tensor(mat, dtype=dtype)


def load_word2vec(embed_file, token2idx, embed_dim, mat_file=None):
    """
    embed_file: 词向量文件
    mat_file: numpy的词向量矩阵，用于快速加载
    """
    if os.path.exists(mat_file):
        embedding_mat = np.load(mat_file)
        return embedding_mat
    else:
        pre_trained = {}  # 词向量文件
        emb_invalid = 0
        for i, line in enumerate(open(embed_file, 'r', encoding='utf-8')):
            line = line.rstrip().split()
            if len(line) == embed_dim + 1:
                pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
            else:
                emb_invalid = emb_invalid + 1

        if emb_invalid > 0:
            print(f'{emb_invalid} invalid lines')

        embedding_mat = np.zeros([len(token2idx), embed_dim])
        count = 0
        for word, idx in token2idx.items():
            if word in pre_trained:
                embedding_mat[idx] = pre_trained[word]
                count += 1
            else:
                embedding_mat[idx] = np.random.normal(size=(embed_dim))  # 没有找到的字用正态分布初始化
        print(f'找到了{count}个字向量, 未找到{len(token2idx) - count}')
        np.save(mat_file, embedding_mat)
        return embedding_mat