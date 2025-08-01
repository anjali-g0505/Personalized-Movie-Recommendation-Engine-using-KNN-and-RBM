# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 23:33:05 2025

@author: Anjali Gangotri
"""

from surprise import AlgoBase

class HybridAlgorithm(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms # List of individual recommendation algorithms to combine.
        self.weights = weights # Corresponding weights for each algorithm.
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset) # Initialize the base algorithm with the training set.
        
        # Iterate through each component algorithm and train it using the provided trainset.
        for algorithm in self.algorithms:
            algorithm.fit(trainset)
                
        return self # Return the fitted hybrid algorithm.

    def estimate(self, u, i):
        # This method predicts a rating for a given user u and item i by combining predictions.
        
        sumScores = 0 # Initialize sum of weighted scores.
        sumWeights = 0 # Initialize sum of weights.
        
        # Iterate through each algorithm and its corresponding weight.
        for idx in range(len(self.algorithms)):
            # Get prediction from the current algorithm and multiply by its weight.
            sumScores += self.algorithms[idx].estimate(u, i) * self.weights[idx]
            sumWeights += self.weights[idx] # Add the current algorithm's weight to the total.
            
        # Return the final weighted average prediction.
        return sumScores / sumWeights