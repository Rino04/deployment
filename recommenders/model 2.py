import pandas as pd
import numpy as np
import pickle
import copy
import keras
from keras.models import load_model
import surprise
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
products_df = pd.read_csv('resources/data/product.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')

from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings_df, test_size=0.2)

n_users, n_product = len(ratings_df.userId.unique()), len(ratings_df.productId.unique())
n_latent_factors = 3

n_latent_factors_user = 5
n_latent_factors_product = 8

product_input = keras.layers.Input(shape=[1],name='Item')
product_embedding = keras.layers.Embedding(n_product + 1, n_latent_factors_product, name='product-Embedding')(product_input)
product_vec = keras.layers.Flatten(name='Flattenproducts')(product_embedding)
product_vec = keras.layers.Dropout(0.2)(product_vec)


user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
user_vec = keras.layers.Dropout(0.2)(user_vec)


concat = keras.layers.merge.concatenate([product_vec, user_vec],name='Concat')
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(200,name='FullyConnected')(concat)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(concat)
dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

    
result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)

model = keras.Model([user_input, product_input], result)
model.compile(optimizer='adam',loss= 'mean_absolute_error')
history = model.fit([train.userId, train.productId], train.rating, validation_split=0.1, epochs=25, verbose=0)
model.save('keras_model.h5')
    
def prediction_item(item_id):
   
    # Data preprosessing
    # model=load_model('keras_model.h5')
    # load_df = ratings_df
    # predictions = []
    # all_users=ratings_df['userId']
    # for ui in all_users:
    #     predictions.append(model.predict(item_id,ui, verbose = False))
    # return predictions

    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    predictions = []
    for ui in model.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_products(product_list):
    
    # Store the id of users
    id_store=[]
    for i in product_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each product with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store
  
def collab_model(product_list,top_n=3):
    

    indices = pd.Series(products_df['title'])
    product_ids = pred_products(product_list)
    df_init_users = ratings_df[ratings_df['userId']==product_ids[0]]
    for i in product_ids :
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
    # Getting the cosine similarity matrix
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