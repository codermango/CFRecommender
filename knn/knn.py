#-*- coding: utf-8 -*-
import numpy as np
import operator
import os


def file2matrix(filepath):
    datasets_path = os.path.join('..', 'datasets')
    ratings_path = os.path.join(datasets_path, filepath, 'ratings.csv')

    # get userid list(labels)
    user_list = []
    with open(ratings_path) as ratings_file:
        rating_list = ratings_file.readlines()
        num_of_user = len(rating_list)
        for i in range(1, num_of_user):
            userid = rating_list[i].split(',')[0]
            if userid not in user_list:
                user_list.append(userid)
    
    print user_list




def knn(input_array, training_dataset, labels, k):
    training_dataset_size = training_dataset.shape[0]
    diff_matrix = np.tile(input_array, (training_dataset_size, 1)) - training_dataset
    square_diff_matrix = diff_matrix ** 2
    square_distance = square_diff_matrix.sum(axis=1)
    distances = square_distance ** 0.5
    sorted_distance_indicies = distances.argsort()
    print distances
    print sorted_distance_indicies

    # selete k least distance point
    print sorted_distance_indicies[:k]
    return sorted_distance_indicies[:k]



# def classify(inputPoint,dataSet,labels,k):
#     dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数
#     #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
#     diffMat = tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
#     sqDiffMat = diffMat ** 2                    #差值矩阵平方
#     sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
#     distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
#     sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表
#     #选择距离最小的k个点
#     classCount = {}
#     for i in range(k):
#         voteIlabel = labels[ sortedDistIndicies[i] ]
#         classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
#     #按classCount字典的第2个元素（即类别出现的次数）从大到小排序
#     sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
#     return sortedClassCount[0][0]



if __name__ == "__main__" :

    file2matrix("ml-latest")


    # dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    # labels = ['A', 'A', 'B', 'B'] 
    # X = np.array([1.2, 1.1])  
    # Y = np.array([0.1, 0.1])
    # k = 3
    # labelX =  knn(X, dataset, labels, k)
    # labelY =  knn(Y, dataset, labels, k)
    # print "Your input is:", X, "and classified to class: ", labelX
    # print "Your input is:", Y, "and classified to class: ", labelY