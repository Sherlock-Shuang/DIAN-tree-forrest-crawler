import numpy as np
import matplotlib.pyplot as plt


def createdata():
    """
      从 iris.data 文件中读取数据，处理类别标签，转换为 numpy 数组并打乱顺序
      :return: 处理后的数据数组和特征标签列表
      """
    try:
        #打开文件读取数据
        with open("./iris.data", "r") as f:
            read = f.read().split('\n')
        data = []
        for line in read:
            if line:
                #分割每行数据
                line = line.split(',')
                #将类别标签转换为数值
                if line[-1] == 'Iris-setosa':
                    line[-1] = 0.0
                elif line[-1] == 'Iris-versicolor':
                    line[-1] = 1.0
                else:
                    line[-1] = 2.0
                #将每行数据转换为浮点数列表
                line = [float(x) for x in line]
                data.append(line)
        #特征标签列表
        labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
        #转换为 numpy 数组
        data = np.array(data)
        #打乱数据顺序
        np.random.shuffle(data)
        return data, labels
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        return None, None


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


def bestsplitpoint(data):
    """
    选择最优分割特征和最优分割点
    :param data: 现存数据
    :return: 最优分割特征索引和最优分割点
    """
    best_gini = float("inf")  # 全局最小基尼指数
    best_feature = -1  # 最佳特征索引
    best_point = None  # 最佳分割点

    # 遍历每个特征
    for i in range(data.shape[1] - 1):
        # 获取当前特征的所有唯一值（跳过重复值）
        points = np.unique(data[:, i])
        # 对连续特征，候选分割点改为相邻值的平均（更精细）
        # 例如特征值 [1,2,4] 生成候选点 [1.5, 3.0]
        if len(points) > 1:
            split_points = (points[:-1] + points[1:]) / 2
        else:
            continue  # 如果特征值全部相同，跳过

        # 遍历每个候选分割点
        for point in split_points:
            # 分割数据
            left_mask = data[:, i] < point
            right_mask = data[:, i] >= point
            left_subset = data[left_mask]
            right_subset = data[right_mask]

            if left_subset.size == 0 or right_subset.size == 0:
                continue

            # 计算加权基尼指数
            left_gini = calcgini(left_subset)
            right_gini = calcgini(right_subset)
            total_gini = (len(left_subset) / len(data)) * left_gini + (len(right_subset) / len(data)) * right_gini

            # 更新全局最优
            if total_gini < best_gini:
                best_gini = total_gini
                best_feature = i
                best_point = point

    return best_feature, best_point  # 返回全局最优的特征和分割点

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



class Node:
    #节点对象
    def __init__(self,gini,num_samples,index,most_class,bestpoint):
        self.gini=gini
        self.index=index
        self.num=num_samples
        self.bestpoint=bestpoint
        self.most_class=most_class
        self.left =None
        self.right =None

class DecisionTree:
    #决策树对象
    def __init__(self,max_depth=3,min_split=2):
        self.max_depth=max_depth
        self.min_split=min_split
        self.root=None

    def growtree(self,data,depth=0):
        num_samples = data.shape[0]
        gini=calcgini(data)
        index,selfpoint= bestsplitpoint(data)
        class_count=[np.sum(data[:,-1]==i) for i in range(3)]
        most_class=np.argmax(class_count)
        node=Node(gini,num_samples,index,most_class,selfpoint)
        if depth<self.max_depth and num_samples>=self.min_split:
            best_feature, best_point = bestsplitpoint(data)
            if best_feature == -1:  # 无法继续分裂
                return node

            # 根据 best_feature 和 best_point 分割数据
            left_data = data[data[:, best_feature] < best_point]
            right_data = data[data[:, best_feature] >= best_point]
            if left_data.shape[0]>0 and right_data.shape[0]>0:
                node.left=self.growtree(left_data,depth=depth+1)
                node.right = self.growtree(right_data, depth=depth + 1)
        return node

#预测一个样本情况
    def predict_one(self, sample, node):
        if node.left is None and node.right is None:
            #叶子节点，返回样本数最多的类别
            return node.most_class
        if sample[node.index] <node.bestpoint:
            return self.predict_one(sample, node.left)
        else:
            return self.predict_one(sample, node.right)

    def predict(self, test_data):
        predictions = []
        for sample in test_data:
            prediction = self.predict_one(sample, self.root)
            predictions.append(prediction)
        return np.array(predictions)



data,labels=createdata()
traindata=data[:int(data.shape[0]*0.75)]#将数据随机分成训练数据和测试数据
testdata=data[int(data.shape[0]*0.75):]
a=testdata[:,-1]


# 构建决策树
tree = DecisionTree()
tree.root = tree.growtree(traindata)

# 进行预测
predictions = tree.predict(testdata)
true_labels = testdata[:, -1]

# 计算准确率
general=true_labels.shape[0]
boo=predictions==true_labels
yes=np.sum(boo)
accuracy=yes/general
print(f"这个模型的准确率是{accuracy}")




