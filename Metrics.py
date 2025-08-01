# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 22:55:57 2025

@author: Anjali Gangotri
"""

import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=4.0): #only recommending 10 movies wiht minimum 4 rating
        topN = defaultdict(list) #dictionary that contains list: {'UserID': [ratings]}
#It creates a default dictionary named topN which adds a list as the value. defaultdict is a class from the collections module in Python.

        for userID, movieID, actualRating, estimatedRating, _ in predictions:#predictions consists of  userID, movieID, actualRating, estimatedRating, details
            if (estimatedRating >= minimumRating):#create a minimumm passing threshold
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)#sort in descending order by estimated rating
            topN[int(userID)] = ratings[:n]# keeps only the top n items for each userID in the topN dictionary.

        return topN #returns a dictionary that maps user ID to ratings
    
# topN = {
#     userID1: [(movieID1, estimatedRating1), (movieID2, estimatedRating2), ... up to N],
#     userID2: [(movieID3, estimatedRating3), ...],
#     ...
#   } 

    def HitRate(topNPredicted, leftOutPredictions):# we are using leave one out cross validation, so leftOutPredictions is the test data left out of training
        hits = 0 
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) : #get total number of leftOut movies present int he top n for 
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total
    
# leftOutPredictions = [
#     (userID1, movieID1, actualRating1, estimatedRating1, _),
#     (userID2, movieID2, actualRating2, estimatedRating2, _),
#     ...
# ]

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0): #example: 100 users, for 30 users, the left out movie is below rating cutoff. Thus for 70 users, what is the hit rate.
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:#for a particualr userID is the movie that is left out for that user(that has rating greater than cutoff) i present in the list
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions): #ratings(5,4.5,4.3) wise hits for each
        hits = defaultdict(float) #dictionary where values are float
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1 #{5: 7, 4.5:6, 4.3: 3...}

            total[actualRating] += 1 #{5: 40, 4.5:20, 4.3: 15...}

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating]) #example: 5: 7/40, 4.5: 6/20, 4.3: 3/15...

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions): #pretty understandable youre measuring reciprocal of hit rank, rank is calcualted by itrating thru a loop
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]: #basically finds out which rank(1-10) is the left out rating at
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) : #sums all those hitRanks up
                summation += 1.0 / hitRank

            total += 1

        return summation / total #ARHR is basically the sum of reciprocals of rank / total users

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold): #if there is even oen good rec by a user then the user is a hit
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo): #very intensive computationally since numebr of pairs for 1 user is 10C2 or 9+8+7+6... which is n(n+1)/2 *number of users. Thus its better to sample your data 
        n = 0 # total number of item-pairs considered
        total = 0 #Sums up all similarity scores between recommended item pairs.
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2) # Gets all unique pairs from the user’s top-N recommended movies. Item-item similarity matrix
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1)) #surprise uses internal IDs for similarity computation 
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2] #Keep a running total of all pairwise similarities.
                total += similarity# total of similarities of all pairs
                n += 1

        S = total / n
        return (1-S) #very diverse is 1 and not diverse is 0

    def Novelty(topNPredicted, rankings): #rankings is a dictionary of popularity rankings of every item as a parameter
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
