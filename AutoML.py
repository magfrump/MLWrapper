# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:59:14 2015

@author: mitchellowen
"""

class AutoMLClassifier:
    
    def __init__(self,maxtime=10):
        '''No arguments, because otherwise it wouldn't be a black box :3
        might add an optional metric argument, first possible values of accuracy, precision, or recall'''
        self.classifiers = []
        #sets up a list for classifiers
        from sklearn.tree import DecisionTreeClassifier
        self.classifiers.append({'clf':DecisionTreeClassifier(),'accuracy':[],'expectation':1,'traintime':0})
        #start with a simple decision tree classifier
        self.maxclassind = 0
        self.maxclassacc = 0
        #set the index and accuracy of the best classifier
        self.types = ['DT']
        self.maxtime=maxtime
        #keep track of whether trees, linear regression have been used
        
    def fit(self,features,labels):
        from time import time
        t0 = time()
        chunksize = 100
        stasistime = 0
        from sklearn.metrics import accuracy_score
        #keep track of how much time is left
        #do a train/test split for CV
        if len(self.classifiers)==1:
            self.classifiers[0]['clf'].fit(features[:chunksize],labels[:chunksize])
            self.classifiers[0]['accuracy'].append(accuracy_score(self.classifiers[0]['clf'].predict(features),labels))
            self.classifiers[0]['expectation'] = self.expect(self.classifiers[0])
            self.traintime = time()-t0
        #start by doing a quick decision tree training and collect data
        while(time()-t0 < self.maxtime-(2*self.traintime) and stasistime<100):
            stasistime+=1
            #while there is time left, choose between training, nearby exploration, and far exploration
            #value of far exploration should be 1-maxclassacc
            #value of near exploration should be difference between maxclassacc and second highest acc
            #value of training should be ex for each classifier
            #this should result in a dynamic where at first more classifiers are added,
            #some are trained slightly, then more nearby classifiers are added
            #OR further classifiers are added if the first aren't effective
            #then eventually most time is spent training.
            
            #for now, though, not that.
            from random import randint
            nexttask = randint(0,len(self.classifiers)+2)
            t1 = time()
            if nexttask< len(self.classifiers):
                t = len(self.classifiers[nexttask]['accuracy'])
                t = t% len(labels)/chunksize
                self.classifiers[nexttask]['clf'].fit(features[t*chunksize:(t+1)*chunksize],labels[t*chunksize:(t+1)*chunksize])
                self.classifiers[nexttask]['accuracy'].append(accuracy_score(self.classifiers[nexttask]['clf'].predict(features),labels))
                if self.classifiers[nexttask]['accuracy'][-1]>self.maxclassacc:
                    self.maxclassacc = self.classifiers[nexttask]['accuracy'][-1]
                    self.maxclassind = nexttask
                    stasistime=0
                self.traintime = (self.traintime+time()-t1)/2
            else:
                nexttask -= len(self.classifiers)
                if nexttask==0:
                    from sklearn.tree import DecisionTreeClassifier
                    self.classifiers.append({'clf':DecisionTreeClassifier()})
                elif nexttask==1:
                    from sklearn.linear_model import SGDClassifier
                    self.classifiers.append({'clf':SGDClassifier()})
                self.classifiers[-1]['clf'].fit(features[:chunksize],labels[:chunksize])
                self.classifiers[-1]['accuracy']=[accuracy_score(self.classifiers[-1]['clf'].predict(features),labels)]
                if self.classifiers[-1]['accuracy'][-1]:
                    self.maxclassacc = self.classifiers[-1]['accuracy'][-1]
                    self.maxclassind = len(self.classifiers)-1
                    stasistime=0
                self.classifiers[-1]['expectation'] = self.expect(self.classifiers[-1])
                self.traintime=(time()-t1 + self.traintime)/2
                
        

    def expect(self,classifier):
        '''Takes in a classifier, outputs an expectation, which should be a percent chance
        that training the classifier will give better than current best performance.
        Currently, it is a hack.'''
        tr = len(classifier['accuracy'])
        ex = 0
        for i in range(1,tr):
            ex+= (classifier['accuracy'][i]-classifier['accuracy'][i-1])/((tr-i)**2)
        return ex
        
    def predict(self,features):
        '''returns results from it's highest accuracy classifier'''
        return self.classifiers[self.maxclassind]['clf'].predict(features)