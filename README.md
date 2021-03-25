## NLP_Chinese_down_stream_task
NLP子任务，供学习参考

### 任务1 ：短文本分类  
#### (1).数据集：THUCNews中文文本数据集(10分类)  
  
![](https://github.com/guang11644331/NLP_Chinese_down_stream_task/blob/master/image/cls_data.png)  

#### (2).模型：BERT+FC，Pytorch实现  
#### (3).使用方法：    
预训练模型使用的是中文**BERT-WWM**, 下载地址(https://github.com/ymcui/Chinese-BERT-wwm), 下载解压后放入[**bert_pretrain**]文件夹下，运行“main.py”即可  
#### (4).训练结果：    
  
![](https://github.com/guang11644331/NLP_Chinese_down_stream_task/blob/master/image/bert_res2.png)  
  
___
### 任务2：命名体识别(NER)  
#### (1).数据集：china-people-daily-ner-corpus（中国人民日报数据集）, Tensorflow_cpu >= 2.1  
  
![](https://github.com/guang11644331/NLP_Chinese_down_stream_task/blob/master/image/ner_data1.png)  

#### (2).模型：BiLSTM+CRF  
  
![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_data.png)  
  
使用了中文Wikipedia训练好的100维词向量，运行main.py即可。  
#### (3).训练结果:  
  
  ![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_1.png)  
#### (4).F1-Score结果:  
  
![](https://github.com/guang11644331/NLP_CHN_down_stream_task/blob/master/image/ner_2.png)  
  
___  
### 任务3：文本匹配（语义相似度，Semantic Textual Similarity）  
**TODO**
