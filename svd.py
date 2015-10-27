'''
Version:1.0
Created on 2014-02-25
@Author:Dior
'''

import random
import math
import cPickle as pickle

class SVD():
    def __init__(self, allfile, trainfile, testfile, factorNum=10):
        #all data file
        self.allfile = allfile
        #training set file
        self.trainfile = trainfile
        #testing set file
        self.testfile = testfile
        #get factor number
        self.factorNum = factorNum
        #get user number
        self.userNum = self.getUserNum()
        #get item number
        self.itemNum = self.getItemNum()
        #learning rate
        self.learningRate = 0.01
        #the regularization lambda
        self.regularization = 0.05
        #initialize the model and parameters
        self.initModel()

    #get user number function
    def getUserNum(self):
        all_file = self.allfile
        cnt = 0
        userSet = set()
        for line in open(all_file):
            user = line.split('\t')[0].strip()
            if user not in userSet:
                userSet.add(user)
                cnt += 1
        return cnt
    
    #get item number function
    def getItemNum(self):
        all_file = self.allfile
        cnt = 0
        itemSet = set()
        for line in open(all_file):
            item = line.split('\t')[1].strip()
            if item not in itemSet:
                itemSet.add(item)
                cnt += 1
        return cnt
    
    #initialize all parameters
    def initModel(self):
        self.av = self.average(self.trainfile)
        self.bu = [0.0 for i in range(self.userNum)]
        self.bi = [0.0 for i in range(self.itemNum)]
        temp = math.sqrt(self.factorNum)
        self.pu = [[(0.1 * random.random() / temp) for i in range(self.factorNum)] for j in range(self.userNum)]
        self.qi = [[0.1 * random.random() / temp for i in range(self.factorNum)] for j in range(self.itemNum)]
        # print self.pu
        print "Initialize end.The user number is: %d,item number is: %d,the average score is: %f" % (self.userNum, self.itemNum, self.av)


     #train model  
    def train(self, iterTimes=100):
        print "Beginning to train the model......"
        trainfile = self.trainfile
        preRmse = 10000.0
        for iter in range(iterTimes):
            fi = open(trainfile, 'r')
            #read the training file
            for line in fi:
                content = line.split('\t')
                user = int(content[0].strip()) - 1 #why minus 1, because the user will be index in the following
                item = int(content[1].strip()) - 1
                rating = float(content[2].strip())

                #calculate the predict score
                predict_score = self.predictScore(self.av, self.bu[user], self.bi[item], self.pu[user], self.qi[item])
                #the delta between the real score and the predict score
                eui = rating - predict_score
                
                #update parameters bu and bi(user rating bais and item rating bais)
                self.bu[user] += self.learningRate * (eui - self.regularization * self.bu[user])
                self.bi[item] += self.learningRate * (eui - self.regularization * self.bi[item])
                for k in range(self.factorNum):
                    temp = self.pu[user][k]
                    #update pu,qi
                    self.pu[user][k] += self.learningRate * (eui * self.qi[user][k] - self.regularization * self.pu[user][k])
                    self.qi[item][k] += self.learningRate * (temp * eui - self.regularization * self.qi[item][k])
                #print predict_score,eui
            #close the file
            fi.close()

            #calculate the current rmse
            curRmse = self.test(self.av, self.bu, self.bi, self.pu, self.qi)
            print "Iteration %d times,RMSE is : %f" % (iter+1, curRmse)
            if curRmse>preRmse:
                break
            else:
                preRmse=curRmse
        print "Iteration finished!"

    #test on the test set and calculate the RMSE
    def test(self, av, bu, bi, pu, qi):
        testfile = self.testfile
        rmse = 0.0
        cnt = 0
        fi = open(testfile)
        for line in fi:
            cnt += 1
            content = line.split('\t')
            user = int(content[0].strip()) - 1
            item = int(content[1].strip()) - 1
            score = float(content[2].strip())
            pscore = self.predictScore(av, bu[user], bi[item], pu[user], qi[item])
            rmse += math.pow(score - pscore, 2)
        fi.close()
        return math.sqrt(rmse / cnt)

    #calculate the average rating in the training set
    def average(self, filename):
        result = 0.0
        count = 0
        for line in open(filename):
            count += 1
            score = float(line.split('\t')[2].strip())
            result += score
        return result / count

    #calculate the inner product of two vectors
    def innerProduct(self, v1, v2):
        result = 0.0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def predictScore(self, av, bu, bi, pu, qi):
        pscore = av + bu + bi + self.innerProduct(pu, qi)
        if pscore < 1:
            pscore = 1
        if pscore > 5:
            pscore = 5
        return pscore
    

def split_files(filepath):
    # 70% for training, 30% for testing
    pass



if __name__=='__main__':
    s=SVD("datasets/ml-100k/u.data","datasets/ml-100k/ua.base","datasets/ml-100k/ua.test")
    #print s.userNum,s.itemNum
    #print s.average("data\\ua.base")
    s.train()

    














