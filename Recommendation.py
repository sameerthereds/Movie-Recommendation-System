# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:55:01 2018

@author: sneupane
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Recommendation:
    
    def __init__(self):
        self.ratings_df=pd.read_csv("ml-latest-small/ratings.csv")
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.genome_tag= pd.read_csv("ml-latest-small/genome-tags.csv")
        self.genome_score=pd.read_csv("ml-latest-small/genome-scores.csv")
        self.crew_df = pd.read_csv("ml-latest-small/data.tsv",delimiter="\t",encoding="utf-8")
        self.director_movies_df = pd.read_csv("ml-latest-small/data_director_movies.tsv",delimiter="\t",encoding="utf-8")
        self.links_df = pd.read_csv("ml-latest-small/links.csv")
        self.director_df = self.get_director_data()
       # merge_df =  pd.merge(self.ratings_df, self.movies_df, on = 'movieId')
        #self.movies_df_mean = merge_df.groupby('title')['rating'].mean().sort_values(ascending=False)
        merge_df =  pd.merge(self.movies_df,self.ratings_df, on = 'movieId',how="left")
        self.movies_df_mean = merge_df.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        # recommend movie from genre
        
        
    def update_mean(self):
#        merge_df =  pd.merge(self.ratings_df, self.movies_df, on = 'movieId')
#        self.movies_df_mean = merge_df.groupby('title')['rating'].mean().sort_values(ascending=False)
        merge_df =  pd.merge(self.movies_df,self.ratings_df, on = 'movieId',how="left")
        self.movies_df_mean = merge_df.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        
    
#    def getMoviesFromGenres(self, genrelist):
#        movies_to_recommend = pd.DataFrame(columns=['movieId','title','genres','Release date','mean_ratings'])
#        
#        genre_list= genrelist
#        for items in genre_list:
#            movies_df_genre = self.movies_df.loc[self.movies_df['genres'].str.contains(items)]
#            title = movies_df_genre.title
#            t = self.movies_df_mean.get(title).values
#            t = pd.Series(t)
#            movies_df_genre["mean_ratings"] = t.values
#            movies_df_genre=movies_df_genre.sort_values(by=['mean_ratings'],ascending=False)
#            movies_df_genre = movies_df_genre.head(20)
#            sample_movies = movies_df_genre.sample(5) 
#            movies_to_recommend = movies_to_recommend.append(sample_movies)
#        return movies_to_recommend,sample_movies
    
    def getUserMovieMatrix(self):
#        ratings_df1 = pd.read_csv("ml-latest-small/ratings.csv")
        user_matrix = self.ratings_df.pivot(index = 'userId', columns ='movieId', values='rating').fillna(0)
        users = len(user_matrix.values.tolist())
        movies = set(self.movies_df['movieId'].tolist())
        movies_rated = set(self.ratings_df['movieId'].tolist())
        movies_nr = []
        for movie in movies:
            if movie not in movies_rated:
                movies_nr.append('movie')
        dfToAdd = [[0 for i in range(len(movies_nr))] for j in range(users)]
        newdf= pd.DataFrame(dfToAdd, columns = movies_nr)
        new_user_matrix = user_matrix.join(newdf)
        return new_user_matrix
    
    # append year in the movie dataframe
    def get_movies_year(self):
        total_dates=[]
        movies_taken = self.movies_df.title
        for items in movies_taken:
            if "(" in str(items):
                item=items[::-1].split(")")        
                item = item[1].split("(")        
                if len(item[0]) == 4:
                    total_dates.append((int(item[0][::-1])))
                else:
                    total_dates.append(0)
            else:
                total_dates.append(0)
        return total_dates
    
    
    def getTagFeatureMatrix(self):
        movie_ours = list(self.movies_df.movieId)
        self.genome_score = self.genome_score.loc[self.genome_score["movieId"].isin(movie_ours)]
        genome_score_matrix = self.genome_score.pivot(index='movieId',columns='tagId',values='relevance')
        movie_features =pd.merge(self.movies_df,genome_score_matrix,on="movieId",how="left").fillna(0)
        movie_features=movie_features.drop(["title"],axis=1)
        movie_features=movie_features.drop(["genres"],axis=1)   
        return movie_features
    
    def GetMoviesFromGenres(self, genrelist):
        movies_to_recommend = pd.DataFrame(columns=['movieId','title','genres', 'mean_ratings'])
        
        genre_list= genrelist
        for items in genre_list:
            movies_df_genre = self.movies_df.loc[self.movies_df['genres'].str.contains(items)]
            title = movies_df_genre.movieId
            t2 = self.movies_df_mean.get(title).values
            t2 = pd.Series(t2)
            movies_df_genre["mean_ratings"] = t2.values
            movies_df_genre = movies_df_genre.sort_values(by=['mean_ratings'],ascending=False)
            movies_df_genre = movies_df_genre.head(20)
            sample_movies = movies_df_genre.sample(5) 
            movies_to_recommend = movies_to_recommend.append(sample_movies)
        print(movies_to_recommend)
        return movies_to_recommend
    
    def getGenreFeatureMatrix(self):
        MovieGenreMatrix = []
        for index, row in self.movies_df.iterrows():
            genres = row['genres'].split("|")
            for genre in genres:
                MId = row['movieId']
                title = row['movieId']
                MovieGenreMatrix.append([MId, title, genre, 1])
        header = ['movieId', 'title', 'genre', 'presence']
        dfMovieGenre = pd.DataFrame(MovieGenreMatrix, columns = header)
        genre_matrix = dfMovieGenre.pivot(index='movieId', columns='genre', values='presence').fillna(0)
        return genre_matrix
    
    def merge_all(self):
        date_df = pd.Series(self.get_movies_year())
        self.movies_df['Release date'] = date_df.values
        movie_year = self.movies_df[['movieId','Release date']]
        movie_tag_feature = self.getTagFeatureMatrix()
        dfGenre = self.getGenreFeatureMatrix()
        movie_id_total=self.movies_df.movieId
        dfGenre["movieId"]= movie_id_total.values
        
        # merging genre, year and tag feature
        movies_feature_matrix = pd.merge(movie_year,dfGenre,on='movieId')
        movies_feature_matrix=pd.merge(movies_feature_matrix,movie_tag_feature,on='movieId')
        movies_feature_matrix=movies_feature_matrix.set_index(movie_id_total.values)
        movies_feature_matrix_final=movies_feature_matrix.drop(["movieId"],axis=1)
        return movies_feature_matrix_final
    
    # pca apply
    def pca_implement(self, movies_feature_matrix_final):
        x = movies_feature_matrix_final[movies_feature_matrix_final.columns[:,] ]   
        x = StandardScaler().fit_transform(x)    
        pca = PCA(.95)
        principalComponents = pca.fit_transform(x)
        principalDf=pd.DataFrame(data=principalComponents)
        return principalDf
    
    def generate_user_movie_matrix(self):
        movie_id_total = self.movies_df.movieId
        movies_feature_matrix_final = self.merge_all()
        principalDf = self.pca_implement(movies_feature_matrix_final)
        principalDf=principalDf.set_index(movie_id_total.values)
        # the utility matrix 
        UserMovieMatrix = self.getUserMovieMatrix()  
        UserMovieMatrix = np.nan_to_num(UserMovieMatrix)
        utility_matrix = np.dot(UserMovieMatrix,principalDf)
        utility_matrix=np.nan_to_num(utility_matrix)
        tran = principalDf.as_matrix()
        movie_utility= np.dot(utility_matrix,np.transpose(tran))
#        ,index=np.array(range(1, 621))
        movie_utility_df = pd.DataFrame(data=movie_utility, columns=movie_id_total.values)
        return movie_utility_df
    
    def recommendMovie(self, user, n):
        movies = self.movies_df['movieId'].tolist()
        row = self.generate_user_movie_matrix().loc[int(user)-1].tolist()
        movie_dict = {}
        for i in range(len(movies)):
            movie_dict[movies[i]] = row[i]
        user_prof = self.ratings_df.loc[self.ratings_df['userId'] == user]
        movies_rated = user_prof['movieId'].tolist()
        recommendedMovies = []
        sorted_dict = sorted(movie_dict.items(), key=lambda kv: kv[1], reverse=True)
        for movie, value in sorted_dict:
            if movie not in movies_rated:
                recommendedMovies.append(movie)
            if len(recommendedMovies) == n:
                break
        movies_to_recommend = []
        for movie in recommendedMovies:
            recommendedMovieRow = self.movies_df.loc[self.movies_df['movieId'] == movie]
            movieid = recommendedMovieRow['movieId'].item()
            title = recommendedMovieRow['title'].item()
            genres = recommendedMovieRow['genres'].item()
            release_date = recommendedMovieRow['Release date'].item()
            t = self.movies_df_mean.loc[movieid]
#            t = self.movies_df_mean.get(title)
            recommendedMovieRow["mean_ratings"] = t
            movies_to_recommend.append([movieid, title, genres, release_date, t])
        moviesToRecommend = pd.DataFrame(movies_to_recommend, columns=['movieId','title','genres','Release date','mean_ratings'])
        return moviesToRecommend
    
    def get_director_data(self):
        a = self.crew_df.tconst
        b=[]
        for items in a:
            b.append(items[2:])
        self.crew_df = self.crew_df.drop(["tconst"],axis=1)  
        self.links_df = self.links_df.drop(["tmdbId"],axis=1)
        self.crew_df = self.crew_df.drop(["writers"],axis=1)  
        self.crew_df["imdbId"] = b
        self.crew_df["imdbId"] = self.crew_df["imdbId"].apply(int)
        self.links_df["imdbId"] = self.links_df["imdbId"].apply(int)
        director_imdb = pd.merge(self.links_df, self.crew_df, on="imdbId", how="left")
        
        c= self.director_movies_df.nconst
        self.director_movies_df = self.director_movies_df.drop(["nconst"],axis=1)
        self.director_movies_df=self.director_movies_df.drop(["birthYear"],axis=1)
        self.director_movies_df=self.director_movies_df.drop(["deathYear"],axis=1)
        self.director_movies_df=self.director_movies_df.drop(["primaryProfession"],axis=1)
        self.director_movies_df["directors"] = c
        
        director_final_df=pd.merge(director_imdb,self.director_movies_df,on="directors",how="left").fillna(0)
        
        e=director_final_df.knownForTitles    
        k=[]
        for items in e:
            if(items!=0):
                i=items.split(",")
                if(len(i)>1):                
                    j=[i[0][2:],i[1][2:]]
                    k.append(j)
                else:
                    k.append(i[0][2:])
            else:  
                k.append(items)
        m=[]
        n=[]
        for items in k:
            if(items!=0):        
                if(len(items)>1):
                    #print(i[0])
                    m.append(items[0])
                    n.append(items[1])
                else:
                    m.append(items[0])
                    n.append(0)
            else:  
                m.append(items)
                n.append(0)
                
        director_final_df["rec imdb id 1"]=m
        director_final_df["rec imdb id 2"]=n    
        director_final_df["rec imdb id 1"]=director_final_df["rec imdb id 1"].apply(int)
        director_final_df["rec imdb id 2"]=director_final_df["rec imdb id 2"].apply(int)
        director_final_df=director_final_df.drop(["knownForTitles"],axis=1)    
        director_final_df_rec = pd.merge(director_final_df,self.links_df,left_on="rec imdb id 1",right_on="imdbId",how="left").fillna(0)
        director_final_df_rec = pd.merge(director_final_df_rec,self.links_df,left_on="rec imdb id 2",right_on="imdbId",how="left").fillna(0)
        director_final_df_rec=director_final_df_rec.drop(["imdbId_x"],axis=1)
        director_final_df_rec=director_final_df_rec.drop(["imdbId_y"],axis=1)
        director_final_df_rec=director_final_df_rec.drop(["rec imdb id 1"],axis=1)
        director_final_df_rec=director_final_df_rec.drop(["rec imdb id 2"],axis=1)
        director_final_df_rec=director_final_df_rec.drop(["imdbId"],axis=1)
        return director_final_df_rec
    
#    director_df= get_director_data(crew_df,links_df,director_movies_df)
    
    def getPopularMovies(self, movies):
        try:
            popular_movie=[]    
            for movie in movies:
                popularmovie1=[]
                popularmovie2=[]        
                director_df1=self.director_df.loc[self.director_df["movieId_x"]==movie]        
                popularmovie1=list(director_df1.movieId_y)
                popularmovie2=list(director_df1.movieId)        
                if (int(popularmovie1[0])>0 and int(popularmovie2[0])>0):
                    popular_movie.append(popularmovie1[0])
                    popular_movie.append(popularmovie2[0])
                elif (int(popularmovie1[0])>0):
                    popular_movie.append(popularmovie1[0])
                elif (int(popularmovie2[0])>0):
                    popular_movie.append(popularmovie2[0])
            return list(set(popular_movie))
        except:
            return list()
            
    
#pop = getPopularMovies(director_df,[193609])
#    movies_to_recommend = recommendMovie(45, 20)
            
            
            
            
            
    
    
    
    
    
    
    
