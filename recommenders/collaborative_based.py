import pandas as pd
import numpy as np
import pickle
import copy
import streamlit as st
import surprise
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
products_df = pd.read_csv('resources/data/product.csv',sep = ',', index_col=0)
ratings_df = pd.read_csv('resources/data/ratings.csv',sep = ',', index_col=0)

min_rat = ratings_df['rating'].min()
max_rat = ratings_df['rating'].max()

users=ratings_df['userId'].unique()
#import tensorflow as tf
#model=tf.keras.models.load_model('resources/models/my_model.h5')
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(productId):
   
    # Data preprosessing
    reader = Reader(rating_scale = (min_rat,max_rat))
    load_df = Dataset.load_from_df(ratings_df[['userId', 'productId', 'rating']],reader)
    predictions = []
    for ui in users:
        predictions.append(model.predict(iid=productId,uid=ui, verbose = False))
    return predictions

def pred_products(product_list):
    
    # Store the id of users
    id_store=[]
    for i in product_list:
        predictions = prediction_item(productId = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each product with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store
  
def collab_model(product_list,top_n=3):
    indices = pd.Series(products_df['productId'])
    product_ids = pred_products(product_list)
    df_init_users = ratings_df[ratings_df['userId']==product_ids[2]]

    
    #df_init_users=0
    for i in product_ids :
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])

    # count_vec = CountVectorizer()
    # count_matrix = count_vec.fit_transform(df_init_users)
    #Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
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
    # Appending the names of products
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    recommended_products = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen products
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_products.append(list(products_df['title'])[i])
    return recommended_products