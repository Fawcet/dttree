from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
import os
import random
import math


testnum = 50     #测试集样本数量 
test_data = []
test_target = []
train_data =[]
train_target = []


def Dataset():  #数据集分割函数
    iris = load_iris()
    rank = [0]*testnum
    i = 0
    while i < testnum:
        rank[i] = random.randint(0, len(iris.data)-1)
        temp = i
        for j in range(temp):
            if rank[j] == rank[i]:
                i = i-1
        i = i+1
    for i in range(len(iris.data)):
        if i in rank:
            test_data.append(iris.data[i])
            test_target.append(iris.target[i])
        else:
            train_data.append(iris.data[i])
            train_target.append(iris.target[i])


def Ent(target):  #信息熵
    p = [0]*3
    ent = 0
    for i in range(3):
        p[i] = target.count(i)/len(target)
    for i in range(3):
        if p[i] == 0:
            ent = ent
        else:
            ent = ent - p[i]*math.log(p[i], 2)
    return ent


def Gain(data, target, t):  #信息增益
    target1 = []
    target2 = []
    for i in range(len(data)):
        if data[i] < t:
            target1.append(target[i])
        else:
            target2.append(target[i])
    gain = Ent(target) - (len(target1)/len(target))*Ent(target1) - (len(target2)/len(target))*Ent(target2)
    return gain


def trans(data):
    num = len(data)
    data_copy = [[0] * num, [0] * num, [0] * num, [0] * num]
    for i in range(4):
        for j in range(num):
            data_copy[i][j] = data[j][i]
    return data_copy


def TreeGenerate(data_old, target):  #决策树生成
    array = [0]
    array1 = [0]
    label = []
    feature = []
    threshold = []
    zuozishu = []
    youzishu = []
    zuocnt = []
    youcnt = []
    k = 0
    m = 0
    data = trans(data_old)
    array[0] = data
    array1[0] = target
    while (1):
        zuo = 0
        you = 0
        if k > m:     #算法出口：若扩展结点均遍历完，则break 
            break
        if k == 0:
            if len(list(set(target))) == 1:
                label.append(target[0])
            else:
                label.append(-1)
        if label[k] == -1:
            num = len(array[k])
            max_t = [float('-inf')]*num
            max_Gain = [float('-inf')]*num
            data10, data11, data12, data13 = [], [], [], []
            target1 = []
            data20, data21, data22, data23 = [], [], [], []
            target2 = []
            for i in range(4):    #寻找具有最大信息增益的属性及其阈值 
                temp = sorted(list(set(array[k][i])))
                for j in range(len(temp)-1):
                    t = (temp[j] + temp[j+1])/2
                    if Gain(array[k][i], array1[k], t) > max_Gain[i]:
                        max_Gain[i] = Gain(array[k][i], array1[k], t)
                        max_t[i] = t
            node_feature = max_Gain.index(max(max_Gain))
            node_threshold = max_t[node_feature]
            feature.append(node_feature)
            threshold.append(node_threshold)
            for i in range(len(array1[k])):    #左右子树的分类 
                if array[k][node_feature][i] < node_threshold:
                    data10.append(array[k][0][i])
                    data11.append(array[k][1][i])
                    data12.append(array[k][2][i])
                    data13.append(array[k][3][i])
                    target1.append(array1[k][i])
                    zuo = zuo + 1
                else:
                    data20.append(array[k][0][i])
                    data21.append(array[k][1][i])
                    data22.append(array[k][2][i])
                    data23.append(array[k][3][i])
                    target2.append(array1[k][i])
                    you = you + 1
            m = m + 2
            array.append([data10, data11, data12, data13])
            array.append([data20, data21, data22, data23])
            array1.append(target1)
            array1.append(target2)
            zuozishu.append(m-1)
            youzishu.append(m)
            zuocnt.append(zuo)
            youcnt.append(you)
            label.append(-1)
            label.append(-1)
            if len(list(set(target1))) == 1:
                label[-2] = target1[0]
            if len(list(set(target2))) == 1:
                label[-1] = target2[0]
            if len(data10) == 0:
                m = m - 1
                del array[-2]
                del array1[-2]
                del label[-2]
            if len(data20) == 0:
                m = m - 1
                del array[-1]
                del array1[-1]
                del label[-1]
        else:
            feature.append(-1)
            threshold.append(-1)
            zuozishu.append(-1)
            youzishu.append(-1)
            zuocnt.append(-1)
            youcnt.append(-1)
        k = k + 1
    return zuozishu, youzishu, label, feature, threshold, zuocnt, youcnt


def Testtree(zuozishu, youzishu, label, feature, threshold, data, target):   #测试集测试函数 
    k = 0
    while (1):
        if label[k] == 0 or label[k] == 1 or label[k] == 2:   #算法出口 
            if label[k] == target:
                accuracy = 1
            else:
                accuracy = 0
            break
        if k > len(label):
            accuracy = 0
            break
        if data[feature[k]] < threshold[k]:  #算法核心 
            k = zuozishu[k]
        else:
            k = youzishu[k]
    return accuracy


def Test(zuozishu, youzishu, label, feature, threshold):  #测试函数，对每一个测试样本调用测试函数 
    accuracy = 0
    for i in range(testnum):
        accuracy = accuracy + Testtree(zuozishu, youzishu, label, feature, threshold, test_data[i], test_target[i])
    return accuracy


def Print(accuracy):	#输出函数 
    ratio = accuracy / testnum
    print('测试正确数量:', accuracy)
    print('测试错误数量:', testnum - accuracy)
    print('测试集总数量:', testnum)
    print('正确率：', ratio * 100, '%\n')


def Cut_Tree(accuracy):   #剪枝函数 
    zuozishu, youzishu, label, feature, threshold, zuocnt, youcnt = TreeGenerate(train_data, train_target)
    while(1):
        cnt = 0
        for k in range(len(label)):
            if zuozishu[k] == -1 and youzishu[k] == -1:
                continue
            elif label[zuozishu[k]] == -1 or label[youzishu[k]] == -1:
                continue
            else:
                temp1 = zuozishu[k]          #保存数据 
                temp2 = youzishu[k]
                temp3 = label[zuozishu[k]]
                temp4 = label[youzishu[k]]
                temp5 = label[k]
                if zuocnt[k] >= youcnt[k]:		#剪枝 
                    label[k] = label[zuozishu[k]]
                else:
                    label[k] = label[youzishu[k]]
                label[zuozishu[k]] = -1
                label[youzishu[k]] = -1
                if accuracy <= Test(zuozishu, youzishu, label, feature, threshold):
                    accuracy = Test(zuozishu, youzishu, label, feature, threshold)		
                    cnt = cnt + 1
                    zuozishu[k] = -1
                    youzishu[k] = -1
                else:
                    zuozishu[k] = temp1		#恢复数据 
                    youzishu[k] = temp2
                    label[zuozishu[k]] = temp3
                    label[youzishu[k]] = temp4
                    label[k] = temp5
        if cnt == 0:   #算法出口 
            break
    return accuracy


def Plot_tree():		#画树函数 
    iris = load_iris()
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")  #输出为pdf 


def main():    #主函数 
    Dataset()
    zuozishu, youzishu, label, feature, threshold, zuocnt, youcnt = TreeGenerate(train_data, train_target)
    accuracy1 = Test(zuozishu, youzishu, label, feature, threshold)
    Plot_tree()
    print('剪枝前：')
    Print(accuracy1)
    accuracy2 = Cut_Tree(accuracy1)
    if accuracy2 > accuracy1:
        print('剪枝后：')
        Print(accuracy2)
    else:
        print('不需要剪枝')


main()

