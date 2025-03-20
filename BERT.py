from transformers import BertTokenizer
from transformers import BertModel
import torch
import json

# 从“评论和评分”的jsonl文件提取数据集
f=open("评论和评分 缩减版.jsonl","r",encoding="UTF-8").read()
f=f.split('\n')
data=[]
for line in f:
    line = line.strip()
    if line:
        data.append(json.loads(line))
#划分训练集和测试集对应的评论x和评分y
traindata=data[0:int(len(data)*0.75)]
testdata=data[int(len(data)*0.75):0]
trainx=[]
trainy=[]
testx=[]
testy=[]
for i in traindata:
    trainx.append(i["text"])
    trainy.append(i["point"])
for i in testdata:
    testx.append(i["text"])
    testy.append(i["point"])

# 导入已经训练好的bert-base-chinese,初始化分词器和模型
model_name="bert-base-chinese"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)

tokens=tokenizer(trainx,add_special_tokens=True,padding=True,truncation=True)
input_ids=tokens["input_ids"]
attention_mask=tokens["attention_mask"]
print(input_ids,attention_mask)







