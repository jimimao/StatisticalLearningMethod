"""
@author : jimimao
@function: Adaboost + decesion stump(weaker classifier)
@description
    Using decesion stump as the base learner to train a boosting classifier
"""

import numpy as np

"""
load data
:return
W0: adaboost初始样本权重
"""
def loaddata():
    dataMat= np.array([[0,1,3],[0,3,1],[1,2,2],
                      [1,1,3],[1,2,3],[0,1,2],
                      [1,1,2],[1,1,1],[1,3,1],[0,2,1]])
    dataLabel = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    m,n = dataMat.shape
    W = np.ones(m) / m
    return dataMat,dataLabel,W



class Adaboost:
    def __init__(self):
        self.a = 0

    """
    训练弱分类器
    WeakerClassifier: 构建一个弱分类器，此处用决策树桩，就一个根结点
    :param
    dataMat: 输入数据
    dataLabel: 数据标签
    W: 样本的权重矩阵
    """
    def WeakerClassifier(self,dataMat,dataLabel,W):
        m,n = dataMat.shape  # m 为样本数，n为样本的特征维度
        steps = 50  # 阈值搜索间隔
        minem = float("inf")
        feature = 0 # 用于划分决策树桩的 标签维度 0 ～ n-1
        threshold = 0 # threshold
        flag = "lt"  # lt: 小于threshold 认为是正例1，rt: 大于threshold 认为是正例1
        for i in range(n):
            data = dataMat[:,i]
            featureMax = data.max()
            featureMin = data.min()
            interval = (featureMax-featureMin) / steps
            for step in range(0,steps):
                thre = featureMin + step * interval
                # :lt
                classlabel = np.ones(m)
                classlabel[data > thre] = -1  # 小于threshold 认为是正例1，标记为lt
                em = self.cal_em(classlabel,dataLabel,W)  # 计算em
                if em < minem:
                    feature,minem,threshold,flag = i,em,thre,"lt"
                # :rt
                classlabel = np.ones(m)
                classlabel[data < thre] = -1  # 大于threshold 认为是正例1，标记为rt
                em = self.cal_em(classlabel, dataLabel, W)
                if em < minem:
                    feature,minem,threshold,flag = i,em,thre,"rt"
            # print(i,em)
        return feature,threshold,flag,minem

    """
    计算分类误差率 em
    """
    def cal_em(self,classlabel,dataLabel,W):
        m = classlabel.shape
        tag = np.ones(m)
        tag[classlabel == dataLabel] = 0  # 标注不相等的维度置1
        em = np.multiply(tag,W)
        return em.sum()

    """
    使用弱分类器预测分类结果
    """
    def WeakerCla(self,dataMat,para):
        m,_ = dataMat.shape
        pre = np.ones(m)
        data =dataMat[:,para['feature']]
        if para['flag'] == 'rt':
            pre[data < para['threshold']] = -1
        else:
            pre[data > para['threshold']] = -1
        return pre

    """
    强分类器
    :param
    W: 初始化样本权重
    numWeakerCla: 弱分类器个数
    """
    def StrongClassifier(self,dataMat,dataLabel,W,numWeakerCla):
        classifier = []
        for i in range(numWeakerCla):
            feature,threshold,flag,em = self.WeakerClassifier(dataMat,dataLabel,W)
            am = 1/2 * np.log( (1- em)/ max(em,1e-31)) # 计算分类器权重
            para = {"am":am,"feature":feature,"threshold":threshold,"flag":flag}
            classifier.append(para)
            G = self.WeakerCla(dataMat,para)
            ex = np.exp(- am * np.multiply(dataLabel,G))
            # 更新W
            Z = np.sum(np.multiply(W,ex))
            W = np.multiply(W ,ex) / Z
        return classifier

    """用强分类器分类"""
    def StrongCla(self,dataMat,dataLabel,classifier):
        m,_ = dataMat.shape
        am = []
        out = []
        for para in classifier:
            out.append(self.WeakerCla(dataMat,para))
            am.append(para['am'])
        am = np.array(am)[:,np.newaxis]
        out = np.array(out)
        pre = np.sign(np.sum(np.multiply(am,out),axis=0))  # 计算强分类器的预测结果
        res = np.zeros_like(dataLabel)
        res[dataLabel == pre] = 1
        return np.sum(res) / m



if __name__ == '__main__':
    dataMat,dataLabel,W0 = loaddata()  #W0 为初始权重
    print(W0)
    cla = Adaboost()
    classifier = cla.StrongClassifier(dataMat,dataLabel,W0,50)
    print(classifier)
    am,feature,threshold,flag = [],[],[],[]
    for item in classifier:
        am.append(item['am'])  # 分类器权重
        feature.append(item['feature'])  # 弱分类器选择的特征分类维度
        threshold.append(item['threshold'])  # feature 对应的阈值
        flag.append(item['flag'])  # lt 表示小于threshold 为1，rt表示大于threshold 为1
    print(am,'\n',feature,'\n',threshold,'\n',flag)
    acc = cla.StrongCla(dataMat,dataLabel,classifier)
    print(acc)

