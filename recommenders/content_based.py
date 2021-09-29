
# Import Libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
products = pd.read_csv('resources/data/product.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
products.dropna(inplace=True)

def data_preprocessing(subset_size):
    
    products['keyWords'] = products['reviews_text'].str.replace('|', ' ')
    # Subset of the data
    products_subset = products[:subset_size]
    return products_subset
  
def content_model(product_list,top_n=10):
    
    # Initializing the empty list of recommended products
    recommended_products = []
    data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the product that matches the title
    idx_1 = indices[indices == product_list[0]].index[0]
    idx_2 = indices[indices == product_list[1]].index[0]
    idx_3 = indices[indices == product_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar products
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)

    # Store product names
    recommended_products = []
    # Appending the names of products
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen products
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_products.append(list(products['title'])[i])
    return recommended_products