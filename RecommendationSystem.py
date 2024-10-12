# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:01:39 2018

@author: RJ
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import pandas as pd
import random
from dash.dependencies import Input, Output, State
import json
import time
from Recommendation import Recommendation as rm

#initializing dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://codepen.io/chriddyp/pen/brPBPO.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
recoClass = rm()
genreList = list()
recommendedMovies = pd.DataFrame()
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Input(id='txtUserID', type="text"),
#        html.Div([html.Button(id="cbGenre-"+str(index)) for index in range(0,10)]),
        html.Div([html.Div(id='hdmid-{}'.format(count)) for count in range(25)]),
        html.H3('Recommended Movies', style={"padding": "20px"}),
        dcc.Link('Refresh', href="/Movies", id="btnRefresh"),
        html.Div([
            html.Div([dcc.Slider(id='sl-{}'.format(count)) for count in range(25)]),
        ], style={"background-color": "#4ABDAC"})
    ], style={"display": "none"}),
    html.Div(id="intermGneres", style={"display": "none"}),
    html.Div(id="intermUserID", style={"display": "none"}),
    html.Div([html.Div([html.Div([html.Span(["X"], className="close", id="spanClose"), html.H2("Congratulation")], className="modal-header"), html.Div(className="modal-body", id="modalBody")], className="modal-content")], id="newIDModal", className="modal"),
    html.Div(id='page-content')
])

#index Page
imgMovie = 'movie.png' 
encoded_imgMovie = base64.b64encode(open(imgMovie, 'rb').read())
imgGenre = 'genre.png' 
encoded_imgGenre = base64.b64encode(open(imgGenre, 'rb').read())
index_page = html.Div([
    html.Div([
        html.Span("Movie Recommendation System", style={"font-family": "fantasy", "font-size": "x-large", "padding": "20px"}),
        dcc.Link('New User', href='/GenreSelection', className="button u-pull-right", style={"background-color": "#F7B733"}),
        dcc.Link('Sign In', href='/Movies', className="button u-pull-right", style={"background-color": "forestgreen"}),
        dcc.Input(
            placeholder='Enter your ID',
            type='number',
            value='',
            className='u-pull-right',
            id='txtUserID'
        )
    ]),
    html.Div([], style={"clear": "both"}),
    html.Div([
        html.Div([
            html.H3("Movies", style={"color": "#F7DC1B"}),
            html.P("Choose from a rich collection of movies. This system assists you in finding the movies you might like. You will also be able to build your custom profile. Next time you come in our system,  we will recommend you the movies you like.")
        ], style={"width":"40%", "float": "left"}),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_imgMovie.decode()), style={"width": "100%"})        
        ], style={"width":"40%", "float": "left"}),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_imgGenre.decode()), style={"width": "100%"})        
        ], style={"width":"40%", "float": "left", "margin":"20px"}),
        html.Div([
            html.H3("Genres", style={"color": "#F7DC1B"}),
            html.P("Some of you might like Action movies. And some of you may say I am into Romantic movies. Do not Worry!! We got you covered. We have a wide range of genres for you to choose your favorite genres and we will recommend you the movies you like.")
        ], style={"width":"40%", "float": "left"}),
    ], style={"height": "79vh", "margin": "20px 0 0 0", "background-color": "#4ABDAC", "padding": "80px 0 0 260px"})
])
#end index Page

#genre selection Page
def GetAllGenres():
    genres = set()
    for s in (recoClass.movies_df['genres'].str.split('|').values):
        genres = genres.union(set(s))
    try:
        genres.remove('(no genres listed)')
    except Exception as ex:
        print(ex)
    return genres

def SelectRandom10Genres():
    global genreList
    genreList = random.sample(GetAllGenres(), 10)

def GenerateDynamicGenres():
    cb_id = 'cbGenre-'
    SelectRandom10Genres()
    global genreList
    component = html.Div(
        [html.Div(html.Button(genre, id=cb_id+str(index), style={"height": "120px", "width": "100%", "border-color": "blanchedalmond", "color": "white"}), style={"padding": "20px", "display": "inline-block", "width": "17%"}) for index, genre in enumerate(genreList)])

    @app.callback(Output('intermGneres', 'children'), [Input(cb_id+str(index), 'n_clicks') for index in range(len(genreList))])
    def UpdateCountGenres(*val):
       lstClicks = list()
       for clicks in val:
           if clicks is not None and clicks%2 != 0:
               lstClicks.append(True)
           else:
                lstClicks.append(False)
       return json.dumps(lstClicks)
   
    @app.callback(Output('btnNext', 'style'), [Input("intermGneres", "children")])
    def UpdateNextAppear(selGenres):
       if selGenres is not None:
           if (sum(1 for g in json.loads(selGenres) if g) != 5):
               return {"display": "none"}
           else:
               return {"background-color": "#F7B733", "margin-left": "20px"}
       else:
           return {"display": "none"}

    for index in range(len(genreList)):
        @app.callback(Output(cb_id+str(index), 'style'), [Input(cb_id+str(index), "n_clicks")])
        def UpdateGenreBtn(n_click):
           if n_click is not None:
               if n_click%2!=0:
                   return {"background-color": "gainsboro", "height": "120px", "width": "100%", "border-color": "blanchedalmond", "color": "white"}
               else:
                   return {"height": "120px", "width": "100%", "border-color": "blanchedalmond", "color": "white"}
           else:
                return {"height": "120px", "width": "100%", "border-color": "blanchedalmond", "color": "white"}
        
    return component

genreSelectionLayout = html.Div([
    html.H3('Select Your five favourite genres', style={"padding": "20px"}),
    html.Div([
        html.Div([GenerateDynamicGenres()]),
        html.Br(),
        dcc.Link("Next", href='/Movies', className="button u-pull-right", id="btnNext", style={"display": "none"})
    ], style={"background-color": "#4ABDAC", "height": "85vh"})
])
#end genre selection Page

#movie recommendation for all users
def DynamicRecommendedMovies(dfRecoMovies, lstDirectorMovies = None):
    lstContent = list()
    count = 0
    for index, movie in dfRecoMovies.iterrows():
        lstContent.append(html.Div([html.Div([movie.movieId], id="hdmid-{}".format(count), style={"display":"none"}), html.Div([html.P(movie.title, style={"color": "#3bace1"}), html.P("Genre: "+movie.genres), html.P("Average user rating: "+str(round(movie.mean_ratings, 2)))], className="movietgr"), dcc.Slider(
                min=0,
                max=5,
                step=0.5,
                marks={i: '{}'.format(i) for i in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]},
                value=0,
                id='sl-{}'.format(count)
            )], className="moviethumbn"))
        count += 1
    
    if(dfRecoMovies.size is None or dfRecoMovies.size <= 0):      
        @app.callback(Output('intermUserID', 'children'), [Input("txtUserID", "value"), Input("spanClose", "n_clicks")]+[Input('sl-{}'.format(count), 'value') for count in range(25)], [State('intermUserID', 'children')]+[State('hdmid-{}'.format(count), 'children') for count in range(25)])
        def UpdateNewUserID(signInUserID, n_clickC, *values):
           valRatings = list(values)[:25]
           userID = list(values)[25]
           userID = str(list(values)[25]).split(",")[1] if userID is not None and userID != "" else ""
           valMovieID = list(values)[26:]

           if(signInUserID is not None and signInUserID != ""):
               return "userID,"+str(signInUserID)
           elif n_clickC is None and len([t for t in valRatings if t is not None and t > 0]) == 1 and (userID is None or userID == ""):
               newUserID = int(recoClass.ratings_df.userId.max()) + 1
               i_r = [(i, r) for i, r in enumerate(valRatings) if r is not None and r > 0][0]
               dfToAppend = pd.DataFrame([ [int(newUserID), valMovieID[i_r[0]][0], i_r[1], time.time()] ], columns=['userId', 'movieId', 'rating', 'timestamp'])
               recoClass.ratings_df = recoClass.ratings_df.append(dfToAppend, ignore_index=True)
               recoClass.ratings_df.to_csv("ml-latest-small\\ratings.csv", index=False)
               recoClass.update_mean()
               return "newUserID,"+str(newUserID)
           elif (userID is not None and userID != ""):
               for r_index, r in enumerate(valRatings):
                   if r is not None and r != "" and float(r) > 0:
                       dfTemp = recoClass.ratings_df[(recoClass.ratings_df.userId != int(userID)) | (recoClass.ratings_df.movieId != int(valMovieID[r_index][0]))]
                       dfToAppend = pd.DataFrame([ [int(userID), valMovieID[r_index][0], r, time.time()] ], columns=['userId', 'movieId', 'rating', 'timestamp'])
                       dfTemp = dfTemp.append(dfToAppend, ignore_index=True)
                       dfTemp.to_csv("ml-latest-small\\ratings.csv", index=False)
                       recoClass.ratings_df = dfTemp
                       recoClass.update_mean()
               return "userID,"+str(userID)
           else:
               return ""
        
        @app.callback(Output('modalBody', 'children'), [Input("intermUserID", "children")])
        def UpdateUserCreatedMsg(userID):
           if userID is not None and userID != "":
               userID = str(userID).split(",")
               if userID[0].strip() == "newUserID":
                   return "Your account has been created. Your User ID is " + str(userID[1])
               else: 
                   return ""
           else:
               return ""
        
        @app.callback(Output('newIDModal', 'style'), [Input("modalBody", "children")])
        def UpdateModalVisible(msg):
           if msg is not None and msg != "":
               return {"display": "block"}
           else:
               return {"display": "none"}
    else:
        recommendedMoviesLayout = html.Div([
            html.Div([
                dcc.Input(id='txtUserID', type="text"),
                dcc.Link('Sign In', href='/Movies', className="button u-pull-right", style={"background-color": "forestgreen"})], style={"display": "none"}),
            html.Div([
                html.H3('Recommended Movies', style={"width": "75%", "float": "left"}),
                html.Div([dcc.Link('Refresh', href="/Movies", id="btnRefresh", className="button u-pull-right", style={"color": "black"})], style={"text=align": "center"})
            ]),
            html.Div(style={"clear": "both"}),
            html.Div([
                html.Div(lstContent),
                html.Div([html.H3('Movies from the Director you may like', style={"padding": "20px"})]+[html.Div([html.Div([html.P(recoClass.movies_df[recoClass.movies_df.movieId == dMovieId].title.values[0], style={"color": "#3bace1"}), html.P("Genre: "+ recoClass.movies_df[recoClass.movies_df.movieId == dMovieId].genres.values[0])], className="movietgr")], className="moviethumbn") for dMovieId in lstDirectorMovies] if lstDirectorMovies is not None else [])
            ], style={"background-color": "#4ABDAC"})
        ])

        return recommendedMoviesLayout

rml = html.Div(DynamicRecommendedMovies(pd.DataFrame()))
#end movie recommendation for all users

#main function
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')],
              [State('intermGneres', 'children'),
               State('txtUserID', 'value'),
               State('intermUserID', 'children')])
def RunSystem(pathname, selGenres, userID, oldUserID):
    if pathname == "/GenreSelection":
        return genreSelectionLayout
    elif pathname == "/Movies":
        userID = str(oldUserID).split(",")[1] if oldUserID is not None and oldUserID != "" else userID
        if userID is not None and userID != "":
            userID = int(userID)
            recommendedMoviesF = recoClass.recommendMovie(userID, 25)
            lstMovieID = list(recommendedMoviesF.movieId)
            director_df = recoClass.getPopularMovies(lstMovieID[:2])
            director_df = None if len(director_df) == 0 else director_df
            rml = html.Div([DynamicRecommendedMovies(recommendedMoviesF, director_df)])
            return rml
        else:
            global genreList
            lstGenre = [genreList[index] for index, genFlag in enumerate(json.loads(selGenres)) if genFlag]
            rml = html.Div([DynamicRecommendedMovies(recoClass.GetMoviesFromGenres(lstGenre))])
            return rml
    else:
        return index_page


if __name__ == '__main__':
    app.run_server(debug=True)