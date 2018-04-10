#coding:utf-8
import numpy as np
import os,sys,csv,string
import queue
import matplotlib.pyplot as plt
class Tool:
    def calcPearson(self,x,y):
        if len(x)!=len(y):
            print("the input list length is not equal")
        else:
            if x==[]:
                return None
            else:
                xMean=np.mean(x)
                yMean=np.mean(y)
                sum=0
                for i in range(len(x)):
                    sum+=(x[i]-xMean)*(y[i]-yMean)
                covXY=sum/(len(x))
                return covXY/(np.std(x)*np.std(y));
    def getPearsonClusterResult(self,trainDataFilePath):
        PATH=trainDataFilePath
        filenames=os.listdir(path=PATH)
        data=[]
        for filename in filenames:
            fullPathName=PATH+"\\"+filename
            with open(fullPathName,"r") as fd:
                csv_data=csv.reader(fd)
                tci=[rows[1] for rows in csv_data]
                tci=tci[1:]
                tci=[float(num) for num in tci]
                data.append(tci)
        return [data,filenames,self.cluster(data,0.85)]
    def getDataAndFilenames(self,testDataFilePath):
        PATH = testDataFilePath
        filenames = os.listdir(path=PATH)
        data = []
        for filename in filenames:
            fullPathName = PATH + "\\" + filename
            with open(fullPathName, "r") as fd:
                csv_data = csv.reader(fd)
                tci = [rows[1] for rows in csv_data]
                tci = tci[1:]
                tci = [float(num) for num in tci]
                data.append(tci)
        return [data,filenames]
    def allVisited(self,visited):
        if(np.sum(visited)==len(visited)):
            return 1
        else :
            return 0
    def allGtP(self,data,P,goal,li):
        for i in range(len(li)):
            if(self.calcPearson(data[goal],data[li[i]])) <= P:
                return False
        return True
    def cluster(self,data,P):
        pointSet=[]
        visited=[0 for i in range(len(data))]
        for i in range(len(data)):
            if self.allVisited(visited)==1:
                return pointSet
            if len(pointSet)==0:
                pointSet.append([i])
            else:
                for j in range(len(pointSet)):
                    li=pointSet[j]
                    if self.allGtP(data,P,i,li)==True:
                        pointSet[j].append(i)
                        visited[i]=1
                        break
                if visited[i]!=1:
                    pointSet.append([i])
        return pointSet








