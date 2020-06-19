# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:52:37 2020.

@author: Alex Boivin
"""

import pandas as pd

class processed_wine_data():
    """Filter data from wine_data instance."""
    
    def __init__(self,wine_data,min_number_of_reviews=2):
        """
        Filter data from wine_data instance.

        Parameters
        ----------
        wine_data : TYPE
            DESCRIPTION.
        min_number_of_reviews : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        """"""
        self.wine_data = wine_data
        self.min_number_of_reviews = min_number_of_reviews
        
        # Keep data only from users who have made multiple reviews
        # Count the reviews
        user_reviews_count_df = self.wine_data.review_data.groupby(['Username','WineName','Winery']).size().groupby(['Username']).size()
        print('Total reviews: {}'.format(len(user_reviews_count_df)))
        # Find the users with multiple reviews
        users_with_multiple_interactions_df = user_reviews_count_df[user_reviews_count_df >= self.min_number_of_reviews].reset_index()[['Username']]
        print('Users with at least {min_rev} reviews: {usr_count}'.format(min_rev=self.min_number_of_reviews,usr_count=len(users_with_multiple_interactions_df)))
        # Keep only the reviews from users with multiple reviews
        self.ratings_from_filtered_users_df = self.wine_data.review_data.merge(users_with_multiple_interactions_df,how='right',left_on=['Username'],right_on=['Username'])
        print('Reviews from users with at least {min_rev} reviews: {rev_count}'.format(min_rev=self.min_number_of_reviews,rev_count=len(self.ratings_from_filtered_users_df)))