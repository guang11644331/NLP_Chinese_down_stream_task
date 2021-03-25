from tqdm import tqdm
import torch
import time
from datetime import timedelta

"""
包含方法：
1. data_preprocessing：tokenize, padding, attention_mask
2. 构建DataIterator类: 获取batch
3. get_dataIter
"""


def data_preprocessing(file_path, config):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in tqdm(f):
            if i == 500:
                break
            i += 1
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            token = config.tokenizer.tokenize(text)
            token = ['[CLS]'] + token
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            
            seq_len = len(token)
            maxlen = config.max_len
            attention_mask = []
            if len(token) < config.max_len:
                token_ids += [0] * (maxlen - seq_len)
                attention_mask = [1] * seq_len + [0] * (maxlen - seq_len)
            else:
                attention_mask = [1] * maxlen
                token_ids = token_ids[:maxlen]
            data.append((token_ids, attention_mask, int(label)))

        return data


class DataIterator():
    def __init__(self, data, batch_size, device):
        """
        data: list[data1, data2, data3...]
        """
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.batch_idx = 0  # 第几批
        self.n_batches = len(data) // batch_size  # 一共有几批，取整
        self.residue = False  # 是否有剩余
        if len(data) % batch_size != 0:
            self.residue = True

    # 根据当前数据改
    def _to_tensor(self, data):
        token_ids = torch.LongTensor([line[0] for line in data]).to(self.device)
        attention_mask = torch.LongTensor([line[1] for line in data]).to(self.device)
        label = torch.LongTensor([line[2] for line in data]).to(self.device)
        return ((token_ids, attention_mask), label)

    def __next__(self):
        # 处理最后的剩余
        if self.residue and self.batch_idx == self.n_batches:
            batch = self.data[self.batch_idx * self.batch_size: len(self.data)]
            self.batch_idx += 1
            batch = self._to_tensor(batch)
            return batch
        # 全部处理完后退出
        elif self.batch_idx > self.n_batches:
            self.batch_idx = 0
            raise StopIteration
        # 一批一批的返回数据
        else:
            batch = self.data[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
            self.batch_idx += 1
            batch = self._to_tensor(batch)
            return batch
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue is True:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_dataIter(file_path, config):
    """
    获取数据迭代器
    """
    data_set = data_preprocessing(file_path, config)
    data_Iter = DataIterator(data_set, config.batch_size, config.device)

    return data_Iter

def get_time_dif(start_time):
    """
    获取已经使用的时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))