# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:52:37 2020.

@author: Alex Boivin
"""

import pandas as pd
import numpy as np
import random
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
# from surprise.model_selection import LeaveOneOut, KFold
from surprise.model_selection import RandomizedSearchCV, cross_validate
import time
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.recommender.surprise.surprise_utils import predict, compute_ranking_predictions
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var,\
            map_at_k, ndcg_at_k, precision_at_k, recall_at_k, get_top_k_items)

    
class hyper_tune():
    """Use surprise RandomizedSearchCV to tune hyperparameters."""
    
    def __init__(self,data_ml,min_n_ratings=2):
        # self.data = combined_processed_wine_data
        # self.data_ml = Dataset.load_from_df(self.data, reader=Reader(rating_scale=(1,5)))
        self.data_ml = data_ml
        self.min_n_ratings = min_n_ratings
        
    def __call__(self,min_n_ratings=2):
        
        # Seperate data into A and B sets for unbiased accuracy evaluation
        raw_ratings = self.data_ml.raw_ratings
        # shuffle ratings
        random.shuffle(raw_ratings)
        # A = 90% of the data, B = 10% of the data
        threshold = int(.9 * len(raw_ratings))
        A_raw_ratings = raw_ratings[:threshold]
        B_raw_ratings = raw_ratings[threshold:]
        # make data_ml the set A
        data_ml = self.data_ml
        data_ml.raw_ratings = A_raw_ratings  
        # search grid
        param_grid = {'n_factors': [50,100,150],'n_epochs': [30,50,70], 'lr_all': [0.002,0.005,0.01],'reg_all':[0.02,0.1,0.4,0.6]}
        gs = RandomizedSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=4)#LeaveOneOut(n_splits=10,min_n_ratings=self.min_n_ratings))
        # fit
        start_time = time.time()
        gs.fit(data_ml)
        search_time = time.time() - start_time
        print("Took {} seconds for search.".format(search_time))
        # best RMSE score
        print(gs.best_score['rmse'])
        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])
        
        # get resulting algorithm with tunned parameters
        algo = gs.best_estimator['rmse']
        
        # retrain on the whole set A
        trainset = data_ml.build_full_trainset()
        algo.fit(trainset)
        
        # Compute biased accuracy on A
        predictions = algo.test(trainset.build_testset())
        print('Biased accuracy,', end='   ')
        accuracy.rmse(predictions)
        
        # Compute unbiased accuracy on B
        # make data_ml the set B
        testset = data_ml.construct_testset(B_raw_ratings)
        predictions = algo.test(testset)
        print('Unbiased accuracy,', end=' ')
        accuracy.rmse(predictions)
        
        return(algo)
    
class wine_recomender():
    """Vivino collaborative filtering recomender system."""
    
    def __init__(self,processed_wine_data,tune=False):
        
        self.tune = tune
        self.data = processed_wine_data.combined_ratings_from_filtered_users_data
        self.data_ml = Dataset.load_from_df(self.data, reader=Reader(rating_scale=(1,5)))
        # self.data['Wine'] = self.data[['Winery','WineName']].apply(' - '.join,axis=1)
        # self.data = self.data[['Username','Wine','Rating']]
        
        # cross validation
        if self.tune:
            tunner = hyper_tune(self.data_ml)
            tunned_algo = tunner()
            # cross-validate with 4 folds corresponding to a 75/25 split
            cross_validate(tunned_algo, self.data_ml, measures=['RMSE', 'MAE'], cv=4, verbose=True)
        algo = SVD()
        cross_validate(algo, self.data_ml, measures=['RMSE', 'MAE'], cv=4, verbose=True)
            
        # Use reco_utils with surprise to do final train, test, and evaluation
        # stratified split so that the same set of users will appear in both training and testing data sets since some users have few ratings
        train, test = python_stratified_split(self.data,ratio=0.75,filter_by="user",col_user='Username',col_item='Wine')
        train_set = Dataset.load_from_df(train, reader=Reader(rating_scale=(1, 5))).build_full_trainset()
        
        if self.tune:
            start_time = time.time()
            tunned_algo.fit(train_set)
            train_time = time.time() - start_time
            print("Took {} seconds for tunned training.".format(train_time))
            
            start_time = time.time()
            tunned_predictions = predict(tunned_algo,test,usercol='Username', itemcol='Wine')
            test_time = time.time() - start_time
            print("Took {} seconds for tunned testing.".format(test_time))
            
            start_time = time.time()
            tunned_all_predictions = compute_ranking_predictions(tunned_algo, train, usercol='Username', itemcol='Wine',remove_seen=True)
            test_time = time.time() - start_time
            print("Took {} seconds for tunned  all predictions testing.".format(test_time))
            
        start_time = time.time()
        algo.fit(train_set)
        train_time = time.time() - start_time
        print("Took {} seconds for un-tunned training.".format(train_time))
        
        start_time = time.time()
        predictions = predict(algo,test,usercol='Username', itemcol='Wine')
        test_time = time.time() - start_time
        print("Took {} seconds for un-tunned testing.".format(test_time))
        
        start_time = time.time()
        all_predictions = compute_ranking_predictions(algo, train, usercol='Username', itemcol='Wine',remove_seen=True)
        test_time = time.time() - start_time
        print("Took {} seconds for un-tunned all predictions testing.".format(test_time))
        
        if self.tune:
            eval_rmse = rmse(test, tunned_predictions, col_user='Username', col_item='Wine', col_rating='Rating')
            eval_mae = mae(test, tunned_predictions, col_user='Username', col_item='Wine', col_rating='Rating')
            eval_rsquared = rsquared(test, tunned_predictions, col_user='Username', col_item='Wine', col_rating='Rating')
            eval_exp_var = exp_var(test, tunned_predictions, col_user='Username', col_item='Wine', col_rating='Rating')
            
            k=5
            eval_map = map_at_k(test, tunned_all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
            eval_ndcg = ndcg_at_k(test, tunned_all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
            eval_precision = precision_at_k(test, tunned_all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
            eval_recall = recall_at_k(test, tunned_all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
            
            print('Tunned evaluations:')
            
            print("RMSE:\t\t%f" % eval_rmse,
                  "MAE:\t\t%f" % eval_mae,
                  "rsquared:\t%f" % eval_rsquared,
                  "exp var:\t%f" % eval_exp_var, sep='\n')
            
            print('----')
            
            print("K = {}".format(k),
                  "MAP@K:\t%f" % eval_map,
                  "NDCG@K:\t%f" % eval_ndcg,
                  "Precision@K:\t%f" % eval_precision,
                  "Recall@K:\t%f" % eval_recall, sep='\n')
        
        eval_rmse = rmse(test, predictions, col_user='Username', col_item='Wine', col_rating='Rating')
        eval_mae = mae(test, predictions, col_user='Username', col_item='Wine', col_rating='Rating')
        eval_rsquared = rsquared(test, predictions, col_user='Username', col_item='Wine', col_rating='Rating')
        eval_exp_var = exp_var(test, predictions, col_user='Username', col_item='Wine', col_rating='Rating')
        
        k=5
        eval_map = map_at_k(test, all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
        eval_ndcg = ndcg_at_k(test, all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
        eval_precision = precision_at_k(test, all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
        eval_recall = recall_at_k(test, all_predictions, col_user='Username', col_item='Wine', col_rating='Rating', col_prediction='prediction', k=k)
        
        print('Un-tunned evaluations:')
        
        print("RMSE:\t\t%f" % eval_rmse,
              "MAE:\t\t%f" % eval_mae,
              "rsquared:\t%f" % eval_rsquared,
              "exp var:\t%f" % eval_exp_var, sep='\n')
        
        print('----')
        
        print("K = {}".format(k),
              "MAP@K:\t%f" % eval_map,
              "NDCG@K:\t%f" % eval_ndcg,
              "Precision@K:\t%f" % eval_precision,
              "Recall@K:\t%f" % eval_recall, sep='\n')
        
class processed_wine_data():
    """Filter data from wine_data instance."""
    
    def __init__(self,wine_data,min_number_of_reviews=3):
        """
        Filter data from wine_data instance.

        Parameters
        ----------
        wine_data : TYPE
            DESCRIPTION.
        min_number_of_reviews : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        """
        # Parameters
        self.wine_data = wine_data
        self.min_number_of_reviews = min_number_of_reviews
        
        # Attributes
        clean_results = self._clean_data()
        self.num_scraped_reviews = clean_results[0]
        self.num_cleaned_reviews = clean_results[1]
        self.num_wines = clean_results[2]
        self.num_unique_users = clean_results[3]
        self.num_users_with_multiple_interactions = clean_results[4]
        self.num_ratings_from_filtered_users = clean_results[5]
        self.cleaned_review_data = clean_results[6]
        self.ratings_from_filtered_users_data = clean_results[7]
        self.combined_ratings_from_filtered_users_data = clean_results[8]
        
        
    def _clean_data(self):
        """
        Clean review data.
        
        Removes nans and blank user names, 
        avaeraging reviews for users who rated the same wine multiple
        times, and removing anonymous reviews. Then only keep reviews from
        users with at least min_number_of_reviews.

        Returns
        -------
        total_scraped_reviews : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        wine_count : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        cleaned_review_data_df : TYPE
            DESCRIPTION.
        ratings_from_filtered_users_df : TYPE
            DESCRIPTION.

        """
        total_scraped_reviews = len(self.wine_data.review_data)
        print('Total scraped reviews: {}'.format(total_scraped_reviews))
        # Keep data only from users who have made multiple reviews
        # average ratings for users who rated same wine multiple times. also removes nans and blank names
        cleaned_review_data_df = self.wine_data.review_data.groupby(['Username','WineName','Winery'],as_index=False).mean()
        # remove anonymous reviews
        anonymous_index_vals = cleaned_review_data_df[cleaned_review_data_df['Username'] == 'Vivino User'].index
        cleaned_review_data_df.drop(anonymous_index_vals,inplace=True)
        print('Total cleaned reviews: {}'.format(len(cleaned_review_data_df)))
        # count number of wines
        wine_count = len(cleaned_review_data_df.groupby(['WineName','Winery']).size())
        print('Total wines: {}'.format(wine_count))
        # count the unique users
        user_reviews_count_df = cleaned_review_data_df.groupby(['Username','WineName','Winery']).size().groupby(['Username']).size()
        print('Total unique users: {}'.format(len(user_reviews_count_df)))
        # find the users with multiple reviews
        users_with_multiple_interactions_df = user_reviews_count_df[user_reviews_count_df >= self.min_number_of_reviews].reset_index()[['Username']]
        print('Users with at least {min_rev} reviews: {usr_count}'.format(min_rev=self.min_number_of_reviews,usr_count=len(users_with_multiple_interactions_df)))
        # keep only the reviews from users with multiple reviews
        ratings_from_filtered_users_df = cleaned_review_data_df.merge(users_with_multiple_interactions_df,how='right',left_on=['Username'],right_on=['Username'])
        print('Reviews from users with at least {min_rev} reviews: {rev_count}'.format(min_rev=self.min_number_of_reviews,rev_count=len(ratings_from_filtered_users_df)))
        # combine winery and winename columns into a single wine column
        combined_ratings_from_filtered_users_df = ratings_from_filtered_users_df
        combined_ratings_from_filtered_users_df['Wine'] = combined_ratings_from_filtered_users_df[['Winery','WineName']].apply(' - '.join,axis=1)
        combined_ratings_from_filtered_users_df = combined_ratings_from_filtered_users_df[['Username','Wine','Rating']]
        
        
        
        return total_scraped_reviews, len(cleaned_review_data_df), wine_count,\
            len(user_reviews_count_df), len(users_with_multiple_interactions_df),\
            len(ratings_from_filtered_users_df), cleaned_review_data_df,\
            ratings_from_filtered_users_df, combined_ratings_from_filtered_users_df
            