import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

st.title("Movie Recommender System")


def get_movie_recommendation(movie_name):
    movie_list = movie_features_df[movie_features_df.index.str.contains(movie_name,regex=False)]
    if len(movie_list):
        movie_idx = movie_list.index
        #print(movie_idx, movie_name)
        distances, indices = model_knn.kneighbors(movie_features_df.loc[movie_idx, :].values.reshape(1, -1), n_neighbors=6)

        movie_rec = []

        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(movie_idx))
            else:
                print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]],
                                                               distances.flatten()[i]))
                movie_rec.append(movie_features_df.index[indices.flatten()[i]])
        return movie_rec


movie_features_df = pickle.load(open('movie_features_df.pkl', 'rb'))
#movie_features_df_matrix = pickle.load(open('movie_features_df_matrix.pkl', 'rb'))
movies_dict = pickle.load(open('movies_data.pkl', 'rb'))
movies_data = pd.DataFrame(movies_dict)
# csr_data = pickle.load(open('csr_data.pkl', 'rb'))
# final_dataset = pickle.load(open('final_dataset.pkl', 'rb'))

# csr_data = csr_matrix(final_dataset.values)
# knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# knn.fit(csr_data)
movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)

selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    #movies_data['title'].values
    movie_features_df.index
)
names = get_movie_recommendation(selected_movie_name)
if st.button('Show Recommendation'):
    names = get_movie_recommendation(selected_movie_name)

    # display with the columns
    col1= st.columns(1)

    st.text(names[0])
    st.text(names[1])
    st.text(names[2])
    st.text(names[3])
    st.text(names[4])

