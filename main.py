import streamlit as st
import streamlit.components.v1 as stc

#Load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
#Loas our Dataset
@st.cache
def load_data(data):
	df=pd.read_csv(data, index_col=0)
	return df

#Vectorize
@st.cache
def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat=count_vect.fit_transform(data)
	#get cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat

#Recommendation sys
@st.cache
def get_recommendation(title, cosine_sim_mat,df,num_of_rec=5):
	#indices of product
	product_indices=pd.Series(df.index, index=df['name']).drop_duplicates()
	#indices of course
	idx=product_indices[title]

	#loop into the cosine matr for that index
	sim_scores = list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	selected_product_indices =[i[0] for i in sim_scores[1:]]
	selected_product_scores =[i[0] for i in sim_scores[1:]]

	#Get the dataframe and title
	result_df=df.iloc[selected_product_indices]
	result_df['similarity_score']=selected_product_scores
	final_recommender_products=result_df[['name','reviews_rating','reviews_username','similarity_score']]
	return final_recommender_products.head(num_of_rec)
@st.cache
def search_term_not_found(term,df):
	result_df = df[df['name'].str.contains(term)]
	return result_df

def main():
	st.title('Product Recommendation App')

	menu = ['Home','Recommend','About']
	choice = st.sidebar.selectbox('Menu', menu)

	#Load the data
	df=load_data('data/df_clean.csv')

	if choice =='Home':
		st.subheader('Home')
		st.dataframe(df.head(10))

	elif choice == "Recommend":
		st.subheader('Recommend Products')
		cosine_sim_mat =vectorize_text_to_cosine_mat(df['name'])
		search_term=st.text_input('Search')
		num_of_rec = st.sidebar.number_input("Number of", 4,30,5)
		if st.button('Recommend'):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.beta_expander("Results as Json"):
						results_json = results.to_dict('index')
						st.write(results_json)
					for row in results.iterrows():
						rec_title = row[1][0]
						rec_rating=row[1][1]
						rec_user=row[1][2]
						rec_score=row[1][3]

					st.write("Name",rec_title)
				except:
					results="Not Found"
					st.warning(results)
					st.info('Suggested Options Include:')
					result_df = search_term_not_found(search_term,df)
					st.dataframe(result_df)

				
				#
	else:
		st.subheader("About")
		st.text('Property of teamstars')

if __name__ == '__main__':
	main()