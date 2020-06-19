# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:52:37 2020.

@author: Alex Boivin
"""

import pandas as pd

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
        """"""
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
        
        
    def _clean_data(self):
        """
        Clean review data by removing nans and blank user names, 
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
        # average ratings for users who rated same wine multiple times
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
        
        return total_scraped_reviews, len(cleaned_review_data_df), wine_count,\
            len(user_reviews_count_df), len(users_with_multiple_interactions_df),\
            len(ratings_from_filtered_users_df), cleaned_review_data_df,\
            ratings_from_filtered_users_df