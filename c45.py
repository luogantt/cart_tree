#coding:utf-8

from math import log
import operator
import treePlotter
import pandas as pd
import numpy as np
from collections import Counter
def read_dataset(filename):
    """
    年龄段：0代表青年，1代表中年，2代表老年；
    有工作：0代表否，1代表是；
    有自己的房子：0代表否，1代表是；
    信贷情况：0代表一般，1代表好，2代表非常好；
    类别(是否给贷款)：0代表否，1代表是
    """
    fr=open(filename,'r')
    all_lines=fr.readlines()   #list形式,每行为1个str
    #print all_lines
    labels=['年龄段', '有工作', '有自己的房子', '信贷情况'] 
    #featname=all_lines[0].strip().split(',')  #list形式
    #featname=featname[:-1]
    labelCounts={}
    dataset=[]
    for line in all_lines[0:]:
        line=line.strip().split(',')   #以逗号为分割符拆分列表
        dataset.append(line)
    return dataset,labels

def read_testset(testfile):
    """
    年龄段：0代表青年，1代表中年，2代表老年；
    有工作：0代表否，1代表是；
    有自己的房子：0代表否，1代表是；
    信贷情况：0代表一般，1代表好，2代表非常好；
    类别(是否给贷款)：0代表否，1代表是
    """
    fr=open(testfile,'r')
    all_lines=fr.readlines()
    testset=[]
    for line in all_lines[0:]:
        line=line.strip().split(',')   #以逗号为分割符拆分列表
        testset.append(line)
    return testset

#计算信息熵
def jisuanEnt(dataset):
    
    
    pdn=np.array(dataset)
    
#    pdd=pd.DataFrame(dataset)
    
    ct=Counter(pdn[:,-1])
    
    ct1=pd.Series(ct)
    
    ct2=ct1/ct1.sum()
    
    entro=-ct2*np.log2(ct2)
    
    return entro.sum()
#划分数据集
def splitdataset(dataset,axis,value):
    
    pdd=pd.DataFrame(dataset)
    pdd1=pdd[pdd[axis]==value]
    
    pdd2=pdd1.drop(axis,axis=1)
    
    return pdd2.values
'''
选择最好的数据集划分方式
ID3算法:以信息增益为准则选择划分属性
C4.5算法：使用“增益率”来选择划分属性
'''
#

#C4.5算法
def C45_chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1
    baseEnt=jisuanEnt(dataset)
    bestInfoGain_ratio=0.0
    bestFeature=-1
    for i in range(numFeatures): #遍历所有特征
        featList=[example[i]for example in dataset]  
        uniqueVals=set(featList) #将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt=0.0
        IV=0.0
        for value in uniqueVals:     #计算每种划分方式的信息熵
            subdataset=splitdataset(dataset,i,value)
            p=len(subdataset)/float(len(dataset))
            newEnt+=p*jisuanEnt(subdataset)
            IV=IV-p*log(p,2)
        infoGain=baseEnt-newEnt
        if (IV == 0): # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV                   #这个feature的infoGain_ratio    
        print(u"C4.5中第%d个特征的信息增益率为：%.3f"%(i,infoGain_ratio))
        if (infoGain_ratio >bestInfoGain_ratio):          #选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature


def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont={}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote]=0
        classCont[vote]+=1
    sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCont[0][0]


def C45_createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    C45Tree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return C45Tree 



def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

if __name__ == '__main__':
    filename='dataset.txt'
    testfile='testset.txt'
    dataset, labels = read_dataset(filename)
    #dataset,features=createDataSet()
    #C4.5决策树
    labels_tmp = labels[:] # 拷贝，createTree会改变labels
    C45desicionTree =C45_createTree(dataset,labels_tmp)
    print('C45desicionTree:\n', C45desicionTree)
    treePlotter.C45_Tree(C45desicionTree)
    testSet = read_testset(testfile)
    print("下面为测试数据集结果：")
    print('C4.5_TestSet_classifyResult:\n', classifytest(C45desicionTree, labels, testSet))
    print("---------------------------------------------")

