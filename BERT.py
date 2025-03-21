from transformers import BertTokenizer
from transformers import BertModel
import torch
import json
import torch.nn as nn
import torch.optim as optim

# 从jsonl文件提取数据集
def getdata(name):
    f=open(name,"r",encoding="UTF-8").read()
    f=f.split('\n')
    data=[]
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))
    #划分对应的评论x和评分y
    x=[]
    y=[]
    for i in data:
        x.append(i["text"])
        y.append(i["point"])
    return x,y

trainx,trainy=getdata("评论和评分 缩减版.jsonl")#训练集
testx,testy=getdata("test.jsonl")#测试集

# 导入已经训练好的bert-base-chinese,初始化分词器和模型
model_name="bert-base-chinese"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)

tokens=tokenizer(trainx,add_special_tokens=True,padding=True,truncation=True,return_tensors="pt",max_length=128)
input_ids=tokens["input_ids"]
attention_mask=tokens["attention_mask"]
# 4. 将数据输入 BERT 模型，提取 CLS 向量
with torch.no_grad():  # 不计算梯度，节省内存
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 提取 CLS 向量

# 5. 定义一个简单的分类器
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # 全连接层

    def forward(self, x):
        return self.fc(x)

# 6. 初始化分类器
input_size = cls_embeddings.shape[1]  # BERT 输出的特征维度
num_classes = 10
classifier = SimpleClassifier(input_size, num_classes)

# 7. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(classifier.parameters(), lr=0.001)  # Adam 优化器

# 8. 将标签转换为张量
trainy = torch.tensor(trainy)-1

# 9. 训练分类器
for epoch in range(10):  # 训练 10 轮
    classifier.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度

    # 前向传播
    logits = classifier(cls_embeddings)
    loss = criterion(logits, trainy)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

testy = torch.tensor(testy, dtype=torch.long) - 1
# 10. 测试分类器
classifier.eval()  # 设置模型为评估模式
with torch.no_grad():
    test_tokens = tokenizer(testx, padding=True, truncation=True, return_tensors="pt")
    test_input_ids = test_tokens["input_ids"]
    test_attention_mask = test_tokens["attention_mask"]

    # 提取测试数据的 CLS 向量
    test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
    test_cls_embeddings = test_outputs.last_hidden_state[:, 0, :]

    # 预测
    test_logits = classifier(test_cls_embeddings)
    test_preds = torch.argmax(test_logits, dim=1)
    correct = (test_preds == testy).sum().item()
    accuracy = correct / len(testy)
    print(f"测试集准确率: {accuracy}")
    print("预测的标签：", test_preds.tolist())







