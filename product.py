
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_product_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

features=st.container()

# Data Loading
title_list = load_product_titles('resources/data/product.csv')
products = pd.read_csv('resources/data/product.csv', sep = ',',index_col=0)
ratings = pd.read_csv('resources/data/ratings.csv', index_col=0)
# App declaration
def main():

    page_options = ["Home","Recommender System"]

    page_selection = st.sidebar.radio("Home", page_options)

    if page_selection == "Home":
        st.write('# Product Recommender Engine')
        st.write('### EXPLORE Datafinity Amazon Consumer Reviews Product')
        st.image('resources/imgs/title.jpeg',use_column_width=True)
        
        st.write('Content Based Filtering: Here we used reviews text and reviews ratings features for the model and computed similarity between user and item by taking cosine similarity between each.')
        st.write('Collaborative Based Filtering: We filtered items of user interest based on user-item similarity. Our best model in which we implemented Pytorch performed well with an RMSE of of 0.82')
        st.write('Hybrid Based Filtering: We combined both content based and collaborative based filtering for this technique.')
        st.subheader('Products Dataset')
        st.dataframe(products.head(10))
        st.subheader('Ratings Dataset')
        st.dataframe(ratings.head(10))
        st.write("A recommender system created by team starts")

    with features:
        if page_selection == "Recommender System":
            # Header contents
            st.write('# Product Recommender Engine')
            st.write('### EXPLORE Datafinity Amazon Consumer Reviews Product')
            st.image('resources/imgs/title.jpeg',use_column_width=True)
            # Recommender System algorithm selection
            sys = st.write("Recommender System")
            
            # User-based preferences
            st.write('### Enter Your Three Favorite Products')

            #sel_col,disp_col=st.columns(2)
            product_1 = st.selectbox('Fisrt Option',title_list[1:15200])
            #ratings = disp_col.selectbox("First Rating",options=[1,2,3,4,5], index=0)

            #sel_col1,disp_col1=st.columns(2)
            product_2 = st.selectbox('Second Option',title_list[20055:20255])
            #rating_2 = disp_col1.selectbox("Second Rating",options=[1,2,3,4,5], index=0)

            #sel_col2,disp_col2=st.columns(2)
            product_3 = st.selectbox('Third Option',title_list[22100:22200])
            #rating_3 = disp_col2.selectbox("Third Rating",options=[1,2,3,4,5], index=0)

            fav_products = [product_1,product_2,product_3]
            #fav_ratings=[ratings,rating_2,rating_3]
            # Perform top-10 product recommendation generation
            if st.button("Recommend"):
                    try:
                        with st.spinner('Crunching the numbers...'):
                            top_recommendations = content_model(product_list=fav_products,
                                                                top_n=10)
                        st.title("We think you'll like:")
                        for i,j in enumerate(top_recommendations):
                            st.subheader(str(i+1)+'. '+j)
                    except:
                        st.error("Oops! Looks like this algorithm does't work.\
                                  We'll need to fix it!")
            # if sys == 'Content Based Filtering':
            #     if st.button("Recommend"):
            #         try:
            #             with st.spinner('Crunching the numbers...'):
            #                 top_recommendations = content_model(product_list=fav_products,
            #                                                     top_n=10)
            #             st.title("We think you'll like:")
            #             for i,j in enumerate(top_recommendations):
            #                 st.subheader(str(i+1)+'. '+j)
            #         except:
            #             st.error("Oops! Looks like this algorithm does't work.\
            #                       We'll need to fix it!")


            # if sys == 'Collaborative Based Filtering':
            #     if st.button("Recommend"):
            #         with st.spinner('Crunching the numbers...'):
            #                 top_recommendations = collab_model(product_list=fav_products,top_n=10)
            #         st.title("We think you'll like:")
            #         for i,j in enumerate(top_recommendations):
            #                 st.subheader(str(i+1)+'. '+j)

                    # try:
                    #     with st.spinner('Crunching the numbers...'):
                    #         top_recommendations = collab_model(product_list=fav_products,
                    #                                            top_n=10)
                    #     st.title("We think you'll like:")
                    #     for i,j in enumerate(top_recommendations):
                    #         st.subheader(str(i+1)+'. '+j)
                    # except:
                    #     st.error("Oops! Looks like this algorithm does't work.\
                    #               We'll working to fix it!")




if __name__ == '__main__':
    main()