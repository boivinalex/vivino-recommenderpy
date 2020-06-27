# vivino-recomenderpy
 A personal project to scrape data from Vivino and use a recommender system to recommend new wines.

A collaborative filtering Single Value Decomposition (SVD) -based wine recommender system to recommend new wines to users based on reviews posted on the wine rating app Vivino.com. 

Model fit is cross-validated using k-fold cross-validation. Model hyperparameters can be automatically tuned using the scikit-surprise package. Tuning aims to improve fit but may yeild worse results (check!). Tuning can aim to improve any of the accuracy metrics used by scikit-surprise. After (optionally) tuning and cross-validation, fit is performed on all data to produce recommendations for all users in the data set.

Data set is obtained by scraping Vivino.com using the selenium package.

---

### Details
#### Data scraping (scrape.py)
This script uses Firefox and requires the appropriate driver (see the [docs](https://selenium-python.readthedocs.io/installation.html#drivers)).

The wine_data class is used to acquire and

Search for a subset of wines on Vivino, scrape wine names and user ratings. Note that not all user ratings are shown on Vivino.com. Ratings with no text for example, do not seem to ever be shown. All scrape country, overall rating, total number of ratings, and price (not currently used in recommendation).

Results are stored in an instance of the wine_data class.

Data can be saved as two .csv files. One with wine name, winery name, user rating (data currently used in recommender). The other with wine name, winery name, country, overall rating, number of ratings, price.

#### Data cleaning (recommend.py)
The processed_wine_data class is used to store cleaned scraped review data.
* Remove blank and NaN user names.
* Some users rate the same wine multiple times. Take the average rating.
* Remove anonymous users.
* Only keep reviews from users who have rated multiple wines in the data set (to avoid cold start). Default is 10 reviews.
* Combine the name of the wine and the name of the winery into a unique wine name for each wine.

Results are stored as an instance of the processed_wine_data class.

#### Make recommendations (recommend.py)
The wine_recommender class is used to store recommendation data.

Recommendations are made by predicting the ratings of un-rated wines using collaborative filtering with SVD and keeping the top-K ratings (controlled by `k_predictions`). Default is 5 recommendations.

* Hyperparameters can be tuned by setting `tune=True`. Tuning metric can be changed using `tune_method`.
* K-fold cross-validation is performed. The number of folds (splits) is controlled by `n_splits`. The default is 5, corresponding to a 80/20 split.
* Predictions are made on all user-wine pairs which are not present in the dataset and `k_predictions` number of recommendations are stored in a pandas `DataFrame`.

Results are stored as an instance of the wine_recommender class.