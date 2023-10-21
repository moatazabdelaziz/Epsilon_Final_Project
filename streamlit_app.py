
# importing libraries
import streamlit as st
import pandas as pd 
import joblib
import sklearn
import xgboost
import category_encoders

# loading saved input features and model
Inputs = joblib.load("input_features.pkl")
Model = joblib.load("best_model.pkl")

# function to make predictions
def prediction(rating, genre, year, month, votes, country, budget, gross, runtime, net_profit):
    # creating a test dataframe with the required columns
    test_df = pd.DataFrame(columns = Inputs)
    # assigning input values to the respective columns
    test_df.at[0 , "rating"] = rating
    test_df.at[0 , "genre"] = ", ".join(genre)
    test_df.at[0 , "year"] = year
    test_df.at[0 , "month"] = month
    test_df.at[0 , "votes"] = votes
    test_df.at[0 , "country"] = country
    test_df.at[0 , "budget"] = budget
    test_df.at[0 , "gross"] = gross
    test_df.at[0 , "runtime"] = runtime
    test_df.at[0 , "net_profit"] = net_profit
    # making prediction using the loaded model
    result = Model.predict(test_df)[0]
    return result

# main function to create the Streamlit app
def main():
    # setting the app title
    st.title("Movie Success Predictor")
    # creating input widgets for user input
    rating = st.radio("Rating" , ['Adults', 'Adults Guidance', 'All Audiance', 'Not Rated'])
    genre = st.multiselect("Genre" , ['Drama', 'Adventure', 'Action', 'Comedy', 'Horror', 'Biography',
       'Crime', 'Other', 'Animation'])
    votes = st.slider("Votes" , min_value = 0 , max_value = 2400000 , value = 0 ,step = 100)
    year = st.slider("Year" , min_value = 1980 , max_value = 2020 , value = 0 ,step = 1)
    month = st.slider("Month" , min_value = 1 , max_value = 12 , value = 0 ,step = 1)
    country = st.selectbox("Country" ,['United Kingdom', 'United States', 'Other', 'Canada', 'France',
       'Germany'] )
    budget = st.slider( "Budget" , min_value = 3000 , max_value = 356000000 , value = 0 , step = 1000)
    gross = st.slider( "Gross" , min_value = 309 , max_value = 356000000 , value = 0 , step = 1000)
    runtime = st.slider( "Runtime" , min_value = 63 , max_value = 366 , value = 0 , step = 5)
    net_profit = st.slider( "Net Profit" , min_value = -158031100 , max_value = 1947484000  , value = -159031100 , step = 10000)
    # predict button
    if st.button("Predict"):
        # calling the prediction function with user inputs
        results = prediction(rating, genre, year, month,  votes, country, budget, gross, runtime, net_profit)
        label = ["Unsuccessful" , "Successful"]
        st.text(f"The Movie will be {label[results]}.")
        
if __name__ == '__main__':
    main()    
    
