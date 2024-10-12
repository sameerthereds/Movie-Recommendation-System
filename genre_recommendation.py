# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:55:01 2018

@author: sneupane
"""

import pandas as pd 

ratings_df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")

r = ratings_df.pivot(index ="userId",columns="movieId",values="rating").fillna(0)

movies_df = movies_df.loc[movies_df['title'] != "Death Note: Desu nôto (2006–2007)"]
total_dates=[]
movies_taken = movies_df.title
for items in movies_taken:
    if "(" in items:
        item=items[::-1].split(")")        
        item = item[1].split("(")        
        if len(item[0]) == 4:
            total_dates.append((int(item[0][::-1])))

movies_df = movies_df.loc[movies_df['title'].str.contains("\(")]
date_df = pd.Series(total_dates)
movies_df['Release date'] = date_df.values

merge_df =  pd.merge(ratings_df,movies_df,on = 'movieId')
movies_df_mean = merge_df.groupby('title')['rating'].mean().sort_values(ascending=False)

genrelist=["Action","Adventure","Comedy","Romance","Horror"]
def getMoviesFromGenres(genrelist):
    movies_to_recommend = pd.DataFrame(columns=['movieId','title','genres','Release date','mean_ratings'])
    
    genre_list= genrelist
    for items in genre_list:
        movies_df_genre = movies_df.loc[movies_df['genres'].str.contains(items)]
        title = movies_df_genre.title
        t = movies_df_mean.get(title).values
        t = pd.Series(t)
        movies_df_genre["mean_ratings"] = t.values
        movies_df_genre=movies_df_genre.sort_values(by=['mean_ratings'],ascending=False)
        movies_df_genre = movies_df_genre.head(20)
        sample_movies = movies_df_genre.sample(5) 
        movies_to_recommend = movies_to_recommend.append(sample_movies)
    return movies_to_recommend,sample_movies

test,sample = (getMoviesFromGenres(genrelist))
movie_year = movies_df[['movieId','Release date']]



genome_tag= pd.read_csv("genome-tags.csv")
genome_score=pd.read_csv("genome-scores.csv")
movie_ours = list(movies_df.movieId)
genome_score = genome_score.loc[genome_score["movieId"].isin(movie_ours)]
genome_score_matrix = genome_score.pivot(index='movieId',columns='tagId',values='relevance')

#mer = pd.merge(movie_year,genome_score_matrix)
mer1 = genome_score_matrix.combine_first(movie_year)


        

    

