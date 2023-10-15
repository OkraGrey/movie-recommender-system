import numpy as np
import pandas as pd
import ast

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits,on='title')

#Extracting required features for the recommender
#genres
#movie_id
#keywords
#title
#overview
#cast
#crew

movies = movies[['title','movie_id','cast','crew','genres','keywords','overview']]
#movies.head(1)

#we have to remove duplicate data
#we have to clean the null entities

movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()

def convert(list_of_dictonaries):
    l=[]
    for i in ast.literal_eval(list_of_dictonaries):#conerting stringed list into list
        l.append(i['name'])
    return l

def convert_cast(obj):
    l=[]
    count =0
    for i in ast.literal_eval(obj):
        if count==3:
            break
        count=count+1
        l.append(i['name'])
    return l

def convert_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['cast']=movies['cast'].apply(convert_cast)
movies['crew']=movies['crew'].apply(convert_director)
movies['overview']=movies['overview'].apply(lambda x: x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies_to_drop =[]
for index,row in movies.iterrows():
    if len(row['genres'])==0 or len(row['keywords'])==0 or len(row['crew'])==0:
        movies_to_drop.append(index)
    
movies.drop(movies_to_drop)

movies['tags']=movies['overview']+movies['crew']+movies['genres']+movies['keywords']+movies['cast']

new_df=movies[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags']= new_df['tags'].apply(lambda x: x.lower())

# VECTORIZING THE DATA
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def get_text_words(text):
    return [i for i in text.split() if i.isalpha()]

def stemming(text):
    y=[]
    for i in get_text_words(text):
        y.append(ps.stem(i))
    return " ".join(y)    

new_df['tags']=new_df['tags'].apply(stemming)

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity = cosine_similarity(vectors)
enumerated_s= enumerate(cosine_similarity)

def recommend(movie_name):
    #Finding index of the movie in our dataframe
    movie_index = new_df[new_df['title']==movie_name].index[0]
    #Finding cosine similarity of that particular indexed_movie
    similarity = cosine_similarity[movie_index]
    movies_to_be_recommended = sorted(enumerate(similarity),reverse=True,key=lambda x: x[1])[1:6]
    return [new_df.iloc[x[0]].title for x in movies_to_be_recommended]

#Testing our Recommender Function
for i in recommend("Avatar"):
    print(i)