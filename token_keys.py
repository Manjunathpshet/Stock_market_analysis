import os
from platform import uname
import tweepy
from tweepy import OAuthHandler
import json

access_token = '1454395339457269761-3SFkymmdjPDmdtXrO31e6o4qb07WIi'
access_token_secret = 'WdytbwowcIrH3p5B1cTepiT1yuafnCaQiSxTEut8HufWx'
consumer_key = 'IzMu4jcqHX5Qn8FTjeVdTZvb2'
consumer_secret = 'BrIkgwwZocpW5wYY7XCSJWnwWfA2mRoY5FcGjU46TkUKc7EsZg'


auth = OAuthHandler(consumer_key , consumer_secret )
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth )
# print('auth')

# cursor = tweepy.Cursor(api.user_timeline , id = 'elonmusk', tweet_mode = 'extended').items(1)
cursor = tweepy.Cursor(api.search_tweets ,q = 'reliance', tweet_mode = 'extended').items(1)
for i in cursor:
    pass
    # print(i.full_text)


