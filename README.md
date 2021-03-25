## NLP_Chinese_down_stream_task
硕士期间自学的NLP子任务，供学习参考

### 任务1：短文本分类  
数据集：THUCNews中文文本数据集
使用模型：BERT+BiLSTM，Pytorch实现  
#### 使用方法：  
预训练模型使用的是中文**BERT-WWM**, 下载地址(https://github.com/ymcui/Chinese-BERT-wwm), 下载解压后放入[**bert_pretrain**]文件夹下，运行“main.py”即可  
#### 训练结果：  
![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/bert_res.png)  
  
### 任务2：命名体识别(NER)  
数据集：china-people-daily-ner-corpus（中国人民日报数据集）, Tensorflow_cpu >= 2.1  
使用模型：BiLSTM+CRF  
  
![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_data.png)  
输入时使用了中文Wikipedia训练好的100维词向量，运行main.py即可。  
#### 训练结果:  
  
  ![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_1.png)  
#### Confusion Matrix:  
  
![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_2.png)  
  
### 任务3：文本匹配（语义相似度，Semantic Textual Similarity）  
**TODO**
