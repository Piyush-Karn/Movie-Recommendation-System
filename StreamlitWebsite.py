import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Functions for Recommendation & Posters  
def fetch_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        full_path = ""
    return full_path

def recommend(movie_title, data, similarity):
    # Given a movie title, return a list of recommended movie titles and their poster URLs.
    
    movie_title_lower = movie_title.lower()
    if movie_title_lower not in data['title'].str.lower().values:
        return [], []

    movie_index = data[data['title'].str.lower() == movie_title_lower].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movie_names = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = data.iloc[i[0]].movie_id
        recommended_movie_names.append(data.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie_names, recommended_movie_posters

# Custom CSS Styling
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/1434278254/vector/blue-and-pink-light-panoramic-defocused-blurred-motion-gradient-abstract-background-vector.jpg?s=612x612&w=0&k=20&c=_KXodNw25trgE0xDe0zFnzNiofFgV50aajKpcI9x_8I=");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-blend-mode: overlay;
}
.css-1d391kg {  
    background-color: rgba(0, 0, 0, 0.6);
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    border: 1px solid #444;
}

h1 {
    color: #FF5733 !important;
}

/* Subtext style */
p {
    color: #D3D3D3;
}

label {
    color: #FFFFFF;
}

div.stButton > button {
    background-color: #DC3545;
    color: #DC3545;
    border: 2px solid #FF6B6B;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

div.stButton > button:hover {
    background-color: #d62839;
    transform: translateY(-2px);
}

div[data-baseweb="select"] {
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    border: 1px solid #444;
    border-radius: 5px;
}

h2, h3 {
    color: #F0F8FF;
}

span.movie-name {
    color: #F0F8FF;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit Web App Layout        
st.title("SmartPicks")
st.markdown("<p>Select a movie from the dropdown to see recommendations!</p>", unsafe_allow_html=True)

# Load preprocessed data and similarity matrix from pickle files
@st.cache_resource
def load_data():
    # Load movies data from a local pickle file
    with open("movies_preprocessed.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Fetch the similarity matrix from an external URL
    similarity_url = "https://drive.usercontent.google.com/download?id=1uvoIqrpWX9ILh7vzYUaqA1IV4iRYGDQA&export=download&confirm=t&uuid=adc73067-e26d-47a5-b4a1-fd6a2c56c0a9"  
    response = requests.get(similarity_url)
    if response.status_code == 200:
        sim = pickle.loads(response.content)
    else:
        st.error("Failed to fetch similarity data.")
        sim = None
    return data, sim

data, similarity = load_data()

# Movie selection dropdown
movie_list = data['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button("Show Recommendation"):
    names, posters = recommend(selected_movie, data, similarity)
    if names:
        st.subheader("Recommended Movies:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.text(names[idx])
                if posters[idx]:
                    st.image(posters[idx])
                else:
                    st.write("Poster not available")
    else:
        st.write("Movie not found. Please try another movie.")
