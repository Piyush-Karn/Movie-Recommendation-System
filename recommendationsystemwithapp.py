import streamlit as st
import numpy as np
import pandas as pd
import ast
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Utility functions for data processing #
# To Extract Genre From the Table
def format_genres(genre):
    Lst = []
    for i in ast.literal_eval(genre):
        Lst.append(i['name'])
    return Lst

# To Extract Top 3 Cast Names From the Table
def format_cast(cast):
    Lst = []
    count = 0
    for i in ast.literal_eval(cast):
        if count < 3:
            Lst.append(i['name'])
        count += 1
    return Lst

# To Extract Director Name from the crew column
def extract_directors(crew):
    Lst = []
    for i in ast.literal_eval(crew):
        if i['job'] == 'Director':
            Lst.append(i['name'])
    return Lst

# To Remove spaces from proper names 
def clean_term(Lst):
    return [i.replace(" ", "") for i in Lst]

# To Stem the words to reduce them to their root form
def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


#    Data Loading and Preprocessing     #
@st.cache_data
def load_and_preprocess_data():
    # Load datasets
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    # Keep only the necessary columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    # Process the columns
    movies["genres"] = movies["genres"].apply(format_genres)
    movies["keywords"] = movies["keywords"].apply(format_genres)
    movies["cast"] = movies["cast"].apply(format_cast)
    movies["crew"] = movies["crew"].apply(extract_directors)

    movies["cast"] = movies["cast"].apply(clean_term)
    movies["crew"] = movies["crew"].apply(clean_term)
    movies["genres"] = movies["genres"].apply(clean_term)
    movies["keywords"] = movies["keywords"].apply(clean_term)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Create a combined tag list
    movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

    # Prepare the final dataframe
    formatted_data = movies.drop(columns=["overview", "genres", "keywords", "cast", "crew"])
    formatted_data["tags"] = formatted_data["tags"].apply(lambda x: " ".join(x))
    formatted_data["tags"] = formatted_data["tags"].apply(lambda x: x.lower())

    # Apply stemming to the tags
    formatted_data["tags"] = formatted_data["tags"].apply(stem)

    return formatted_data

@st.cache_resource
def compute_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    sim = cosine_similarity(vectors)
    return sim

#  Functions for Recommendation & Posters  #
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
    # To Handle case-insensitive matching
    movie_title_lower = movie_title.lower()
    if movie_title_lower not in data['title'].str.lower().values:
        return [], []

    # To Get the index of the movie in the dataframe
    movie_index = data[data['title'].str.lower() == movie_title_lower].index[0]

    # To Compute similarity distances and get the top 5 recommendations (excluding the input movie)
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movie_names = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = data.iloc[i[0]].movie_id
        recommended_movie_names.append(data.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters

#        Streamlit Web App Layout        #

st.title("Personalized Movie Recommender")
st.write("Select a movie from the dropdown to see recommendations!")

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    /*background: linear-gradient(to bottom right, #1E1E2E, #3A0CA3, #03045E);*/
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

</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# To Load data and compute similarity matrix 
formatted_data = load_and_preprocess_data()
similarity = compute_similarity(formatted_data)

# Movie selection dropdown
movie_list = formatted_data['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

# When the user clicks the recommendation button, display results
if st.button("Show Recommendation"):
    names, posters = recommend(selected_movie, formatted_data, similarity)
    if names:
        st.subheader("Recommended Movies:")
        # Create five columns for the five recommended movies
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
