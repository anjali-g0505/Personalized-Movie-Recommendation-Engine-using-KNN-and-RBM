# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 23:32:24 2025

@author: Anjali Gangotri
"""

import os 
import csv
import sys
import re

from surprise import Dataset #get the pre-made movie lens dataset
from surprise import Reader #get the Reader 

from collections import defaultdict
import numpy as np

class MovieLens:

    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    
    def loadMovieLensLatestSmall(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0])) #changing the current working directory to the current script's directory 

        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1) #used to parse a file containing ratings. Such a file is assumed to specify only one rating per line. 
#The fields names, in the order at which they are encountered on a line, seperated by char, or ; in the documentation and skip 1 line
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader) #to use a custom dataset, params are file path and reader

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0]) 
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset

        def getUserRatings(self, user): #works for sorted data only
            userRatings = []
            hitUser = False
            with open(self.ratingsPath, newline='') as csvfile:
                ratingReader = csv.reader(csvfile)
                next(ratingReader)
                for row in ratingReader:
                    userID = int(row[0])
                    if (user == userID):
                        movieID = int(row[1])
                        rating = float(row[2])
                        userRatings.append((movieID, rating))
                        hitUser = True
                    if (hitUser and (user != userID)): #so all the ratings by same user are grouped since it is sorted, so if another is user is found and the hit is True then break
                        break
    
            return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True): #returns sorted list of movieId and ratings count,wiht ratings sorted in descending order so 45 ratings, 30,20,19 ...
        #unpacks tuple into movieId and ratingCount automatically
            rankings[movieID] = rank #rankings consist of most rated movieID to least rated (number)
            rank += 1
        return rankings
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs: #if genre is already present in the dict of genreIDs(stores genre: gendre ID) then obtain genre ID from dict
                        genreID = genreIDs[genre] 
                    else: #else make a new genreID and update maxGenreID by 1
                        genreID = maxGenreID
                        genreIDs[genre] = genreID #genreIDs is a dict of genre: genreID
                        maxGenreID += 1
                    genreIDList.append(genreID) #append all new genres and their IDs to the list
                genres[movieID] = genreIDList# movieID: [Id1,ID2,ID3]
                
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors for content-based filtering, similarities etc 
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$") #searches for a year in brackets in the title
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1) #returns group 1 
                if year:
                    years[movieID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getMovieName(self, movieID): #get name of movie from movie id
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def getMovieID(self, movieName): #get id of movie from movie name
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0