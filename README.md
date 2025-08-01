# Personalized-Movie-Recommendation-Engine-using-KNN-and-RBM
Developed a hybrid movie recommendation system integrating Content-Based Filtering and Collaborative Filtering for enhanced accuracy and assessed the system on metrics like RMSE and MAE, achieving an RMSE score of 0.94 and an MAE score of 0.73 on the MovieLens 100K dataset.

# Project Description
Developed a Personalized Movie Recommendation Engine using a hybrid approach combining Content-Based KNN and RBM. It effectively addresses the cold start problem and significantly enhances prediction accuracy, leveraging movie metadata and user ratings.

# Key Features
Hybrid Recommendation System: Integrates Content-Based Filtering (ContentKNN) and Collaborative Filtering (Restricted Boltzmann Machines - RBM) for robust recommendations.

Cold Start Problem Mitigation: The content-based component assists in recommending items even for new users or new items with limited rating data.

Comprehensive Evaluation: Assesses algorithm performance using a wide array of metrics beyond just accuracy.

Custom Similarity Metrics: ContentKNN uses custom genre, year similarities.

RBM for Collaborative Filtering: Leverages a Restricted Boltzmann Machine for learning complex user-item interactions.

# Performance Metrics
The following metrics were achieved during evaluation:

Algorithm	  RMSE  	MAE    	HR    	cHR    	ARHR  	Coverage	Diversity	  Novelty
RBM	        1.1887	0.9928	0.0000	0.0000	0.0000	0.0000	  0.0000	    0.0000
ContentKNN	0.9375	0.7263	0.0030	0.0030	0.0017	0.9285	  0.5700	    4567.1964
Hybrid	    0.9951	0.8123	0.0000	0.0000	0.0000	0.4441	  0.0974	    1462.7042

# Legend
RMSE (Root Mean Squared Error): Lower values indicate better accuracy.

MAE (Mean Absolute Error): Lower values indicate better accuracy.

HR (Hit Rate): How often a left-out rating is recommended. Higher is better.

cHR (Cumulative Hit Rate): Hit rate for ratings above a certain threshold. Higher is better.

ARHR (Average Reciprocal Hit Rank): Hit rate that takes the ranking into account. Higher is better.

Coverage: Ratio of users for whom recommendations exist. Higher is better.

Diversity: Measures how dissimilar recommended items are. Higher means more diverse.

Novelty: Average popularity rank of recommended items. Higher means more novel.

How to Run
Clone the repository: (Assuming this will be in a Git repo)

# Directions to Clone the Repo
git clone <repository-url>
cd <repository-name>
Install dependencies:

pip install surprise numpy tensorflow
Prepare Data: Ensure the ml-latest-small dataset is in the correct directory (e.g., ../ml-latest-small/ relative to your script, as specified in MovieLens.py). Also, ensure LLVisualFeatures13K_Log.csv is accessible for Mise-en-scene features.

Execute the main script:
python hybridtest.py  
# Dependencies  
Surprise: A Python scikit for building and analyzing recommender systems.
NumPy, Pandas, Scikit: Fundamental package for numerical computing with Python.
TensorFlow: Used for the RBM implementation.
