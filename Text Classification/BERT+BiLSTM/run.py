import time
from pytorch_BERT.optimization import BertAdam
import torch.nn.functional as F
from sklearn import metrics
import torch
import utils
import numpy as np
import os


def training(config, model, train_Iter, dev_Iter):
    start_time = time.time()
    # 启动 BatchNormalization 和 dropout
    model.train()
    # bias, LayerNorm.weight, LayerNorm.bias 这三个参数不需要衰减
    no_decay = ['bias', 'LayerNorm.weight']
    named_params = list(model.named_parameters())
    # 需要权重衰减的参数设为0.01， 不需要设为0.0
    is_decay_params = [
        {'params': [param for name, param in named_params if not any(nd in name for nd in no_decay)], 
         'weight_decay': 0.01}, 
        {'params': [param for name, param in named_params if any(nd in name for nd in no_decay)], 
         'weight_decay': 0.0}]
    optimizer = BertAdam(is_decay_params, lr=config.learning_rate, 
                         warmup=0.05, t_total=config.epochs * len(train_Iter))

    dev_best_loss = float('inf')
    total_batch = 0
    flag = False # early stop 标志

    for epoch in range(config.epochs):
        print('Epoch [{:02}/{:02}]'.format(epoch+1, config.epochs))
        for i, (x_batch, y_batch) in enumerate(train_Iter):
            pred = model(x_batch)
            loss = F.cross_entropy(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:  # 每200次打印效果信息
                train_acc = metrics.accuracy_score(y_batch.detach().cpu().numpy(),
                                                   torch.argmax(pred, 1).detach().cpu().numpy())
                dev_loss, dev_acc = evaluation(config, model, dev_Iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    last_imporved = total_batch  # 记录, 用于计算early stop
                    # 保存模型
                    if not os.path.exists(config.save_path):
                        os.mkdir(config.save_path)
                    torch.save(model.state_dict(), os.path.join(config.save_path, 'bert.ckpt'))
                    saved_flag = '*'
                else:
                    saved_flag = ''
                log = 'Batch:{:04}, Train Loss:{:.4f}, Train Acc:{:.4f}, Dev Loss:{:.4f}, Dev Acc:{:.4f}, Time:{} {}'
                print(log.format(i, loss.item(), train_acc, dev_loss, dev_acc, utils.get_time_dif(start_time), saved_flag))
                model.train()  # 因为上面调用的evaluation()里有mdoel.eval(),所以要重启model.train()
            total_batch += 1
            if total_batch - last_imporved > config.early_stop:
                print('early stop...')
                flag = True
                break
        if flag:
            break


def evaluation(config, model, dev_Iter):
    model.eval()
    
    total_loss = 0
    pred_all = []
    y_all = []
    
    with torch.no_grad():
        for x_batch, y_batch in dev_Iter:
            pred = model(x_batch)
            loss = F.cross_entropy(pred, y_batch)
            total_loss += loss
            pred_all += torch.argmax(pred, 1).detach().tolist()
            y_all += y_batch.detach().tolist()

        acc = metrics.accuracy_score(y_all, pred_all)
    return total_loss / len(dev_Iter), acc


def predction(config, model, test_Iter):
    """
    模型测试
    """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    loss, acc = evaluation(config, model, test_Iter)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)
    print('loss:{:.4f}, acc:{:.4f}'.format(loss, acc))