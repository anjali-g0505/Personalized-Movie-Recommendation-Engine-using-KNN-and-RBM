# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 23:48:12 2025

@author: Anjali Gangotri
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLens import MovieLens
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase): #derived class of AlgoBase class

    def __init__(self, k=40, sim_options={}): #finding the right k is experimental
        AlgoBase.__init__(self) # you're inside a subclass (ContentKNNAlgorithm) and want to call the superclass's method on yourself (self), use:
        self.k = k

    def fit(self, trainset): #called when algorithm.fit() is run in EvaluatedAlgorithm class
        AlgoBase.fit(self, trainset) # inside a subclass (ContentKNNAlgorithm) and want to call the superclass's method on yourself (self), use:

        # Compute item similarity matrix based on content attributes

        # Load up genre vectors for every movie
        ml = MovieLens()
        genres = ml.getGenres() #for each movie - genres are filled with yes or no
        years = ml.getYears()
        mes = ml.getMiseEnScene()
        
        print("Computing content-based similarity matrix...")
            
        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items)) #makes a matrix(initialized with zero) where the number of rows adn columns is equal to num of movies
        
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items): #to not repeat the similarity computation for the same one. Forms the upper triangular matrix
                thisMovieID = int(self.trainset.to_raw_iid(thisRating)) #the internal ids are stored from 0 to the total number of movies
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))#this is done so that we can access the genre in the genre list using raw ids and not int ids
                
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                #mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
                
                #doesnt contain the diagonal tho
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity #*mesSimilarity #instead of aimply multiplying we can also weigh them differently : finalSim = 0.7 * genreSim + 0.3 * yearSim   # weighted average

                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating] #copying to the lower triangular matrix
                
        print("...done.")
                
        return self
    
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0 
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0) #when diff is close to 10 -> 0.36 adn when diff is close to 0 -> 1
        return sim
    
    def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if (mes1 and mes2):
            shotLengthDiff = math.fabs(mes1[0] - mes2[0]) #absolute value of x.
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1]) #There is a conceptual flaw here wher a higher diff shud mean less similarity therefore less val but it actually gives a huge val
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes2[5])
            numShotsDiff = math.fabs(mes1[6] - mes2[6])
            return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsDiff
            # This is better: return (1 - shot_diff) * (1 - color_diff) * (1 - motion_diff) * (1 - light_diff) * (1 - shot_count_diff)
        else: #if one of the lists is empty then thats a missing value so: no data = no similarity
            return 0

    def estimate(self, u, i): #i is hte internal id and u is the internal userid

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)): #checks if the user u exists in the training data.
            raise PredictionImpossible('User and/or item is unkown.') #Python keyword that means: "Throw an exception.", predictionimpossible is specific exception class provided by Surprise.
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]: #dictionary of List of tuples(internal_item_id, rating)
            genreSimilarity = self.similarities[i,rating[0]] #looking for similarities in the matrix
            neighbors.append( (genreSimilarity, rating[1]) ) #rating[0] → internal ID of the item user u has rated,  rating[1] → the actual rating user gave that item
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0 
        for (simScore, rating) in k_neighbors:
            if (simScore > 0): #Here we handle hte 0 value so if i is compared against i it will give 0 which wont be calculated
                simTotal += simScore
                weightedSum += simScore * rating #The formula is: predicted rating= ∑(similarity)/∑(similarity×rating)
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')
j 
        predictedRating = weightedSum / simTotal

        return predictedRating
    