# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 22:55:57 2025

@author: Anjali Gangotri
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from RBM import RBM
from MovieLens import MovieLens

class RBMAlgorithm(AlgoBase):  # This is our RBM-based recommendation algorithm, inheriting from Surprise's AlgoBase.

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self) # Calling the constructor of the parent class (AlgoBase) to initialize it.
        self.epochs = epochs # Number of times the entire training dataset will be passed forward and backward through the RBM.
        self.hiddenDim = hiddenDim # The number of hidden neurons in our RBM. More hidden neurons can learn more complex patterns.
        self.learningRate = learningRate # How much the RBM's weights and biases are adjusted during each training step.
        self.batchSize = batchSize # The number of training examples processed before the RBM's parameters are updated.
        self.ml = MovieLens() # Loading the MovieLens dataset using our custom MovieLens class. This data will be used to get movie titles.
        self.ml.loadMovieLensLatestSmall() 
        self.stoplist = ["sex", "drugs", "rock n roll"] # A list of "stop words" or terms that will cause a movie to be excluded (or "stopped") from the recommendation process.
        
    def buildStoplist(self, trainset):
        self.stoplistLookup = {}
        for iiid in trainset.all_items():
            self.stoplistLookup[iiid] = False
            movieID = trainset.to_raw_iid(iiid)
            title = self.ml.getMovieName(int(movieID))
            if (title):
                title = title.lower()
                for term in self.stoplist:
                    if term in title:
                        print ("Blocked ", title)
                        self.stoplistLookup[iiid] = True    
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        self.buildStoplist(trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        # Create a 3D NumPy array for training data.
        # Dimensions: [users, items, rating values (10 for 0.5-5.0)].
        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            if not self.stoplistLookup[iid]:
                # Adjust rating to be 0-indexed (0 to 9) from 0.5-5.0 scale.
                adjustedRating = int(float(rating)*2.0) - 1
                trainingMatrix[int(uid), int(iid), adjustedRating] = 1 # One-hot encode the rating.
        
        # Flatten to a 2D array: each row is a user, columns are (item_1_rating_0.5, ..., item_N_rating_5.0).
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes.
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        rbm.Train(trainingMatrix) # Train the RBM.

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = rbm.GetRecommendations([trainingMatrix[uiid]]) # Get recommendations from RBM for current user.
            recs = np.reshape(recs, [numItems, 10]) # Reshape into [items, rating_scores].
            
            for itemID, rec in enumerate(recs):
                # Convert raw scores to probabilities using softmax.
                normalized = self.softmax(rec)
                # Calculate predicted rating as the weighted average of rating categories.
                rating = np.average(np.arange(10), weights=normalized)
                # Convert 0-9 indexed predicted rating back to 0.5-5.0 scale.
                self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5
        
        return self


    def estimate(self, u, i):
        # Check if user/item are known; if not, prediction is impossible.
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i] # Retrieve pre-calculated rating.
        
        # If rating is near zero, no valid prediction exists.
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating