import pandas as pd
import ast
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#########################################
# Utility functions for data processing #
#########################################

def format_genres(genre):
    lst = []
    for i in ast.literal_eval(genre):
        lst.append(i['name'])
    return lst

def format_cast(cast):
    lst = []
    count = 0
    for i in ast.literal_eval(cast):
        if count < 3:
            lst.append(i['name'])
        count += 1
    return lst

def extract_directors(crew):
    lst = []
    for i in ast.literal_eval(crew):
        if i['job'] == 'Director':
            lst.append(i['name'])
    return lst

def clean_term(lst):
    return [i.replace(" ", "") for i in lst]

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

#########################################
#    Data Loading and Preprocessing     #
#########################################

def preprocess_data():
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

def compute_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    sim = cosine_similarity(vectors)
    return sim

if __name__ == "__main__":
    # Preprocess the data
    formatted_data = preprocess_data()

    # Compute similarity matrix
    similarity = compute_similarity(formatted_data)

    # Save the preprocessed data and similarity matrix to pickle files
    with open("movies_preprocessed.pkl", "wb") as f:
        pickle.dump(formatted_data, f)

    with open("similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    print("Preprocessing complete. Files 'movies_preprocessed.pkl' and 'similarity.pkl' saved.")
