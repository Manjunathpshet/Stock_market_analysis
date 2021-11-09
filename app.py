from logging import error, exception
from pandas.io.pytables import incompatibility_doc
import streamlit as st
from datetime import date
import time
import pandas as pd
import numpy as np
from numpy import array
import math
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import plotly
from scipy.special import boxcox, inv_boxcox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow import keras
# NLP
import os
from platform import uname
import tweepy
from tweepy import OAuthHandler
import json
import spacy
import re
import string
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


st.set_page_config(layout="wide")
st.title('Stock Forecasting App')

stocks = pd.read_csv('stock_list.csv', index_col= 0)

selected_stock = st.selectbox('Select Stock for prediction', stocks['Symbol'])
st.text('')

@st.cache
def get_data(name, start_date= date(2015,1,1) , end_date= date(2022,12,25)):
        data = yf.Ticker(name).history(start = start_date ,end= end_date)
        return data 

st.write('Choose dates for getting the stock data : ')
START = st.date_input("Starting Date" , date(2015,1,1) ) 
TODAY = st.date_input("End Date" , date.today() )

data_load_state = st.text('Loading data...')
data1 = get_data(selected_stock+ '.NS',START ,TODAY )
data_load_state.text('Loading data... done !')

for i in (stocks[stocks['Symbol'] == selected_stock].loc[:,'Company Name']).index:
    company_name = stocks[stocks['Symbol'] == selected_stock]['Company Name'][i]


try:
    new = np.round((data1.tail(1)['Close'][0]),2)
    old = np.round((data1.tail(2)['Close'][0]),2)
    change = np.round(new - old , 2) # Change in price 
    gain = np.round(((new-old)/new * 100 ),2) # % gain 
    ch = (f'{change} ({gain}%)')
    # 52 weeks range
    high_52 = np.round(data1.Close.iloc[len(data1)-260:].max(),2)
    low_52 = np.round(data1.Close.iloc[len(data1)-260:].min(),2)
    range_52 = (f'{low_52} - { high_52}')

    st.subheader(company_name) # Stock name
    st.metric(label="", value= new , delta= ch ,delta_color="normal") # % gain and stock price
    st.text('')
    st.write('52 Weeks Range : ')
    st.write(range_52) ## 52 weeks range vlaues 
    st.text('')
    st.write('👈 Check the sidebar for stock Analysis and Prediction ! ')
    st.text('')
except IndexError:
    st.write('No data found , Please check the stock name entered :')

else : 
     
    data = data1.copy() ## Taking copy of data

    st.text('')
    st.subheader('Historical data')
    st.write('These are the latest 5 days data')
    st.write(data.tail())

    ## Download data as CSV
    @st.cache
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(data1)
    st.download_button(label="Download data as CSV", data = csv ,file_name='Stock_data.csv', mime='text/csv',)

    st.subheader('Area chart for stock Closing price')
    st.area_chart(data.Close, width=8, height= 500 ) # area chart

    st.sidebar.header('What do you want to do with stock')
    user = st.sidebar.selectbox('SELECT FROM BELOW',('', 'ANALYSE', 'FORECAST' , 'SENTIMENT_ANALYSIS'))
    st.sidebar.text('')

    my_bar = st.sidebar.progress(0)  # Progress bar 
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.subheader('What you can do here :')
    st.sidebar.write('1.  Using forcasting method to predict the future 30 days stock value. ')
    st.sidebar.write('2.  Perform technical analysis using charts . ')
    st.sidebar.write('3.  Perform sentiment analysis on different company. ')
    st.sidebar.write('4.  Get the results for different company stocks. ')

    if user == 'ANALYSE':

        new_data = data.drop(columns = {'Dividends', 'Stock Splits'})
        st.subheader('Let\'s Analyse Stock')
        st.write(new_data.describe())
        st.text('')
        st.text('')
        st.subheader('Line chart for stock Closing price')
        st.line_chart(data.Close, width=8, height= 500 )
        st.text('')
        st.text('')
        st.subheader('MOVING AVERAGE')
        st.markdown('->The moving average (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price.The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks or any time period the trader chooses')
        st.markdown('->The most common applications of moving averages are to identify trend direction and to determine support and resistance levels.')
        st.markdown('->When asset prices cross over their moving averages, it may generate a trading signal for technical traders.')
        st.text('')
        st.text('')
        #Visualizations
        st.subheader("Closing Price vs Time chart with 100MA")  # moving avrg
        ma100 = data.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(ma100, label = 'MA100')
        plt.plot(data.Close , label = 'Closing price')
        plt.legend()
        st.plotly_chart(fig)

        #Visualizations
        st.subheader("Closing Price vs Time chart with 100MA & 200MA")  #  moving avrg
        ma100 = data.Close.rolling(100).mean()
        ma200 = data.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(ma100, label = 'MA100')
        plt.plot(ma200, label = 'MA200')
        plt.plot(data.Close , label = 'Closing price')
        plt.legend()
        st.plotly_chart(fig)


    elif user == 'FORECAST':
        # Dropping the columns which are not relevent 
        data =data.drop(columns = {'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'})

        # Applyig box cox transformation
        close_box, lambda_ = stats.boxcox(data['Close'])
        data['Close'] = close_box

        ### LSTM are sensitive to the scale of the data. so we apply min max scaler
        scaler = MinMaxScaler(feature_range=(0, 1))   # values will be scaled between 0 and 1
        data = scaler.fit_transform(data)

        #Splitting dataset into train and test split

        training_size=int(len(data)*0.80)
        test_size=len(data)-training_size
        train_data,test_data= data[0:training_size,:],  data[training_size:len(data),:]


        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(100, len(train_data)):
            x_train.append(data[i-100:i, 0])
            y_train.append(data[i, 0])

        # Convert the x_train and y_train to numpy arrays 
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        #Create the Stacked LSTM model

        # model=Sequential()
        # model.add(LSTM(100,return_sequences=True,input_shape=(X_train.shape[1],1)))
        # model.add(LSTM(100,return_sequences=True))
        # model.add(LSTM(100))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error',optimizer='adam')

        # model.fit(X_train, y_train, epochs=30, batch_size=18, verbose=10)

        model = keras.models.load_model('model')

        #predicting 100 values, using past 30 from the train data


        inputs = data[len(data) - len(test_data) - 100:]
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        # Create the testing data set
        # Create a new array
        X_test = []
        for i in range(100,inputs.shape[0]):
            X_test.append(inputs[i-100:i,0])
            x_test = np.array(X_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        # Get the models predicted price values 
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        rmse_lstm = np.sqrt(np.mean(np.power((test_data-predictions),2)))
        rmse_lstm = inv_boxcox(rmse_lstm, lambda_)

        # Forcasting 
        x_input=test_data[len(test_data)-100:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        st.text('')
        st.text('')
        st.text('')

        n_days = st.number_input('Insert Day\'s of prediction' , min_value = 7 , max_value = 30)

        # demonstrate prediction for next 30 days
        lst_output=[]
        n_steps=100
        i=0
        while(i<n_days):
        
            if(len(temp_input)>100):
                x_input=np.array(temp_input[1:])
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1


        day_new= data1.index[len(data1)-100:]
        day_pred= pd.date_range(date.today(), periods= n_days , freq='D')

        
        st.subheader(f"Predicting for next {n_days} days")  
        fig = plt.figure(figsize=(11,6))
        plt.plot(day_new,inv_boxcox(scaler.inverse_transform(data[len(data)-100 :]), lambda_ ), label = ' actual Closing price')
        plt.plot(day_pred,inv_boxcox(scaler.inverse_transform((lst_output)), lambda_ ), label = 'Predicted')
        plt.grid(True)
        plt.legend()
        st.plotly_chart(fig)
        st.text('')
        st.write('These are the Forecasted values : ')
        df = pd.DataFrame(inv_boxcox(scaler.inverse_transform((lst_output)), lambda_ ), columns= ['Predicted Price'])
        st.write(df)
        st.text('')
        ## Download forecasted data as CSV
        st.text('')
        csv1 = convert_df(df)
        st.download_button(label="Download data as CSV", data = csv1 ,file_name='Forcasted_data.csv', mime='text/csv',)


        # Confidence interval
        output = inv_boxcox(scaler.inverse_transform((lst_output)), lambda_ )
        conf_interval = stats.norm.interval(0.95,loc = output.mean(),scale = output.std())
        low , high = conf_interval
        # st.write( 'Closing price at 95% confidence interval is:' , np.round(conf_interval, 2))
        st.write('Closing price at 95% confidence interval is:' ,low , high )
        

    elif user == 'SENTIMENT_ANALYSIS' :
        ## authenticating 
        st.text('')
        st.subheader('SENTIMENT ANALYSIS FOR :')
        st.subheader(company_name)
        try:
            import token_keys as tk
            access_token =  tk.access_token                  
            access_token_secret = tk.access_token_secret
            consumer_key = tk.consumer_key
            consumer_secret = tk.consumer_secret

            auth = OAuthHandler(consumer_key , consumer_secret )
            auth.set_access_token(access_token,access_token_secret)
            api = tweepy.API(auth)
            # Key word based search
            number_of_tweets = 500
            tweets = []
            likes = []
            Tweet_time = []
            for i in tweepy.Cursor(api.search_tweets , q = selected_stock, tweet_mode = 'extended').items(number_of_tweets):
                tweets.append(i.full_text)
                likes.append(i.favorite_count)
                Tweet_time.append(i.created_at)
            df_st = pd.DataFrame({'Tweets': tweets , 'Likes': likes , 'Time': Tweet_time})
            st.text('')
            st.write(df_st)
            tweet_data = df_st.copy()

        except Exception as exception :
            logger.debug('Token Expired or accessed more times')
            st.write("Can not perform sentiment analysis for now, Sorry for the inconvenience caused, Try using Forecasting")
        else:
            # Data Cleaning
            st.text('')

            def clean_text(text):
                ''' make text lowercase, removing text in square bracket, removing punctuation and words containing numbers'''
                text = text.lower()
                text = re.sub('\[.*?\]','',text)
                text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
                text = re.sub('\w*\d\w*','',text)
                text = re.sub("[0-9" "]+"," ",text)
                text = re.sub('[‘’“”…�]','',text)
                return text

            clean = lambda x: clean_text(x)

            tweet_data['Tweets'] = tweet_data.Tweets.apply(clean)

            TAG_CLEANING_RE = "@\S+"
            # Remove @ tags
            tweet_data['Tweets'] = tweet_data['Tweets'].map(lambda x: re.sub(TAG_CLEANING_RE, ' ', x))

            # Remove numbers
            tweet_data['Tweets'] = tweet_data['Tweets'].map(lambda x: re.sub(r'\d+', ' ', x))

            # Remove links
            TEXT_CLEANING_RE = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
            tweet_data['Tweets'] = tweet_data['Tweets'].map(lambda x: re.sub(TEXT_CLEANING_RE, ' ', x))

            # Sentiment analysis

            afinn = pd.read_csv('Afinn.csv', sep=',', encoding='latin-1')
            affinity_scores = afinn.set_index('word')['value'].to_dict()
            nlp = spacy.load('en_core_web_sm') 

            # Custom function :score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence.
            sentiment_lexicon = affinity_scores

            def calculate_sentiment(text: str = None):
                sent_score = 0
                if text:
                    sentence = nlp(text)
                    for word in sentence:
                        sent_score += sentiment_lexicon.get(word.lemma_, 0)
                return sent_score
            
            # Applying sentiment calculation to the data set and creating column of same

            tweet_data['sentiment_value'] = tweet_data['Tweets'].apply(calculate_sentiment)
            tweet_data['word_count'] = tweet_data['Tweets'].str.split().apply(len)


            # Creating the category column for the above data

            tweet_data["category"] = tweet_data['sentiment_value']

            tweet_data.loc[tweet_data.category > 0, 'category'] = "Positive"
            tweet_data.loc[tweet_data.category != 'Positive', 'category'] = "Negative"

            tweet_data["category"] = tweet_data["category"].astype('category')

            st.text('')
            st.text('')

            positive = []
            negative = []

            for item in tweet_data.category :
                if item == 'Positive':
                    positive.append(item)
                else:
                    negative.append(item)
            
            # st.write('Tweet\s found : ' , len(tweets))
            # st.write (len(tweets))

    
            st.text('')
        
            if np.abs(len(positive) - len(negative)) > 100 and len(positive) > len(negative) :
                st.write('Sentiment Is Positive!! We can plan to buy some shares :heart_eyes: ')
                st.text('')
                st.write(' These are the top Positive Tweets: ')
                st.text('')
                st.write(tweet_data[tweet_data['sentiment_value']>=2].head())
            elif np.abs(len(positive) - len(negative)) > 100 and len(positive) < len(negative):
                st.write('Sentiment Is Negative!! Lets not buy Shares :disappointed:  ')
                st.text('')
                st.write(' These are the top Negative Tweets: ')
                st.text('')
                st.write(tweet_data[tweet_data['sentiment_value']< 0].head())
            else:
                st.write('Market Sentiment is Neutral !! :confused: ')       

    else  :
        pass
        



