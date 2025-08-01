# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 23:35:58 2025

@author: Anjali Gangotri
"""

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    
    def __init__(self, data, popularityRankings):
        
        self.rankings = popularityRankings
        
        #1
        #Build a full training set for evaluating overall properties
        #build_full_trainset()- converts the entire raw dataset (data) into a surprise Trainset object  users/items converted to internal indices..
        self.fullTrainSet = data.build_full_trainset()
        #contains all the user-item pairs that do not appear in self.fullTrainSet. In other words, for every user, it lists all the items they haven't rated. (can form the testset)
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        #Build a 75/25 train/test split for measuring accuracy
        #training set containing 75% of the rating,  test set containing the remaining 25% of the ratings, is used to measure the model's prediction accuracy (e.g., using RMSE or MAE).
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)
        
        #2
        #Build a "leave one out" train/test split for evaluating top-N recommenders
        #And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1) #only 1 iteration of a ranom value of rating that is left out per user
        for train, test in LOOCV.split(data): #just split the data into test- all ratings that were left out(1 per user) and train- all the other movies and their ratings for each user
            self.LOOCVTrain = train
            #goal :can recommender predict this "left out" item in its top recommendations.
            self.LOOCVTest = test
        
        #contains all (user,item) pairs where user hasnt rated the movie... so for LOOCV, it is all movies that the user hasnt rated + the rating in the testset(which is seemingly unrated by the user)
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
        
        #3
        #Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)
            
    def GetFullTrainSet(self):
        return self.fullTrainSet
    
    def GetFullAntiTestSet(self):
        return self.fullAntiTestSet
    
    def GetAntiTestSetForUser(self, testSubject):
        #(All are lsit of tuples except trainset.ur which is a dict of list of tuples)
        trainset = self.fullTrainSet #List of tuple of (userID, itemID, ratings)
        fill = trainset.global_mean  #global_mean is the average of all ratings in the training set. This is used as a placeholder rating value for the anti-testset 
        anti_testset = [] # empty list that will store all the (user, item, dummy_rating) triplets for items the user hasn’t rated yet.
        u = trainset.to_inner_uid(str(testSubject)) #Converts the raw user ID (testSubject) - Can be str or int and is converted to string to make Surprise’s internal user ID. Internally, Surprise uses integers for speed, so user "1" becomes internal ID 0.
        
        #u-internal user id
        user_items = set([j for (j, _) in trainset.ur[u]]) #trainset.ur[u] - list of tuple of all [(item id, rating)...] for the given user that user has rated
        #user_items - set of all itemID in trainset.ur[u] and _ means that that val is not required so just random var is assigned
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if #trainset.all_items is a range-like object of all internal item IDs
                                 i not in user_items] #i is a part of hte testset only is i is not in user_items
        #anti_testset is a list of tuples of all items user has not rated
        return anti_testset

    def GetTrainSet(self):
        return self.trainSet
    
    def GetTestSet(self):
        return self.testSet
    
    def GetLOOCVTrainSet(self):
        return self.LOOCVTrain
    
    def GetLOOCVTestSet(self):
        return self.LOOCVTest
    
    def GetLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet
    
    def GetSimilarities(self):
        return self.simsAlgo
    
    def GetPopularityRankings(self):
        return self.rankings