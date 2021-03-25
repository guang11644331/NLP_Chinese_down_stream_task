import argparse
import numpy as np
from MyBERT import Config, Model
import utils
import time
import torch
import run
import warnings
warnings.filterwarnings("ignore")


def random_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.seed_all()
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样


parser = argparse.ArgumentParser(description='--BERT分类任务--')
parser.add_argument('--model', type=str, default='BERT', help='model_name')
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model  # 命令行参数
    config = Config()
    random_seed()

    start_time = time.time()
    # 加载数据
    train_Iter = utils.get_dataIter(config.train_path, config)
    dev_Iter = utils.get_dataIter(config.dev_path, config)
    test_Iter = utils.get_dataIter(config.test_path, config)

    print('使用时间：', utils.get_time_dif(start_time))

    # 模型训练
    model = Model(config).to(config.device)
    run.training(config, model, train_Iter, dev_Iter)
    run.predction(config, model, test_Iter)

    


    print()