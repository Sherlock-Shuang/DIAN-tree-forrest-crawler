import numpy as np
import matplotlib.pyplot as plt


def createdata():
    """
      从 iris.data 文件中读取数据，处理类别标签，转换为 numpy 数组并打乱顺序
      :return: 处理后的数据数组和特征标签列表
      """
    try:
        # 打开文件读取数据
        with open("./iris.data", "r") as f:
            read = f.read().split('\n')
        data = []
        for line in read:
            if line:
                # 分割每行数据
                line = line.split(',')
                # 将类别标签转换为数值
                if line[-1] == 'Iris-setosa':
                    line[-1] = 0.0
                elif line[-1] == 'Iris-versicolor':
                    line[-1] = 1.0
                else:
                    line[-1] = 2.0
                # 将每行数据转换为浮点数列表
                line = [float(x) for x in line]
                data.append(line)
        # 特征标签列表
        labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
        # 转换为 numpy 数组
        data = np.array(data)
        # 打乱数据顺序
        np.random.shuffle(data)
        return data, labels
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        return None, None


def changedata(data):
    """
      将数据的每个属性根据平均值分为大小两类
      :param data: 输入的数据数组
      :return: 处理后的数据数组
      """
    ave=np.median(data[:,:-1],axis=0)
    left=np.where(data[:,:-1]>ave,1,0)
    right=data[:,-1].reshape(-1,1)
    data=np.concatenate([left,right],axis=1)
    data=data.astype(int)
    return data


def calcgini(data):
    """
      计算数据集的 CART 基尼指数
      :param dataset: 输入的数据集
      :return: 基尼指数
      """
    general=data.shape[0]
    s=0
    for i in range(3):
        count=np.sum(data[:, -1] == i)
        s+=(count/general)**2
    gini=1-s
    return gini

def calcnewgini(data,i):
    """
    计算按照对应属性分割得到的新gini
    :param data: 原矩阵数据
    :param i: 第i个属性
    :return: 新gini
    """
    newgini = 0.0
    general = data.shape[0]
    for j in range(2):
        subset = data[data[:, i] == j]
        count = subset.shape[0]
        if count==0:
            continue
        newgini += (count / general) * calcgini(subset)
    return newgini

def bestsplitdata(data,n_features=None):
    """
    选择最优分割特征
    :param data: 现存数据
    :return: 最优分割特征索引
    """
    bestindex=-1
    bestgini=float("inf")
    all_features=range(data.shape[1]-1)
    if n_features:
        features=np.random.choice(all_features,size=n_features,replace=False)
    else:
        features=all_features
    for i in features:
        newgini=calcnewgini(data,i)
        if newgini<bestgini:
            bestgini=newgini
            bestindex=i
    return bestindex

class Node:
    #节点对象
    def __init__(self,gini,num_samples,most_class):
        self.gini=gini
        self.index=None
        self.num=num_samples
        self.most_class=most_class
        self.left =None
        self.right =None

class DecisionTree:

    def __init__(self,max_depth=4,min_split=3):
        self.max_depth=max_depth
        self.min_split=min_split
        self.root=None

    def growtree(self,data,depth=0,n_features=None):
        num_samples = data.shape[0]
        gini=calcgini(data)
        class_count=[np.sum(data[:,-1]==i) for i in range(3)]
        most_class=np.argmax(class_count)
        node=Node(gini,num_samples,most_class)
        if depth<self.max_depth and num_samples>=self.min_split:
            index = bestsplitdata(data,n_features)
            if index!=-1:
                node.index=index
                left_data= data[data[:, index] == 0]
                right_data= data[data[:, index] == 1]
                if left_data.shape[0]>0 and right_data.shape[0]>0:
                    node.left=self.growtree(left_data,depth+1,n_features)
                    node.right = self.growtree(right_data, depth + 1,n_features)
        return node

    # 预测一个样本情况
    def predict_one(self, sample, node):

         if node.left is None and node.right is None:
         # 叶子节点，返回样本数最多的类别
             return node.most_class
         if sample[node.index] == 0:
             return self.predict_one(sample, node.left)
         else:
             return self.predict_one(sample, node.right)
    #预测一群
    def predict(self, test_data):
        predictions = []
        for sample in test_data:
            prediction = self.predict_one(sample, self.root)
            predictions.append(prediction)
        return np.array(predictions)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features= n_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_feature = X.shape
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_feature))

        for i in range(self.n_trees):
            # Bootstrap采样（有放回抽样）
            sam = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[sam]
            y_bootstrap = y[sam]
            data = np.c_[X_bootstrap, y_bootstrap]

            #训练决策树,特征随机选择
            tree = DecisionTree(max_depth=self.max_depth, min_split=self.min_samples_split)
            tree.root = tree.growtree(data, n_features=self.n_features)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 多数投票
        return np.array([np.bincount(preds).argmax() for preds in tree_preds.T])

def compute_feature_importance(forest, labels):
    importance = {name: 0 for name in labels}
    for tree in forest.trees:
        stack = [tree.root]
        while stack:
            node = stack.pop()
            if node.index is not None:
                importance[labels[node.index]] += 1
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
    # 归一化
    total = sum(importance.values())
    for v in importance:
        importance[v]/=total
    return importance

def plot_feature_importance(importance):
    names = list(importance.keys())
    values = list(importance.values())
    plt.figure()
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance in Random Forest')
    plt.show()


data,labels=createdata()
data=changedata(data)

Y=data[:,-1].astype(int)
traindata=data[:int(data.shape[0]*0.75)]#将数据随机分成训练数据和测试数据
testdata=data[int(data.shape[0]*0.75):]
trainx=traindata[:,:-1]
trainy=traindata[:,-1]
testx=testdata[:,:-1]
testy=testdata[:,-1]


#训练随机森林
rf=RandomForest(n_trees=100,max_depth=3)
rf.fit(trainx,trainy)
#进行预测
predictions = rf.predict(testx)

#计算准确率
general=testy.shape[0]
boo=predictions==testy
yes=np.sum(boo)
accuracy=yes/general
print(f"这个模型的准确率是{accuracy}")

importance = compute_feature_importance(rf, labels)
plot_feature_importance(importance)


