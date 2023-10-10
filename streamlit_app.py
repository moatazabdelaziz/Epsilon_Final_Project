
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
def prediction(rating, genre, year, votes, country, budget, gross, runtime):
    # creating a test dataframe with the required columns
    test_df = pd.DataFrame(columns = Inputs)
    # assigning input values to the respective columns
    test_df.at[0 , "rating"] = rating
    test_df.at[0 , "genre"] = genre
    test_df.at[0,"year"] = year
    # test_df.at[0 , "score"] = score
    test_df.at[0 , "votes"] = votes
    test_df.at[0 , "country"] = country
    test_df.at[0 , "budget"] = budget
    test_df.at[0 , "gross"] = gross
    test_df.at[0 , "runtime"] = runtime
    # displaying the test dataframe
    st.dataframe(test_df)
    # making prediction using the loaded model
    result = Model.predict(test_df)[0]
    return result

# main function to create the Streamlit app
def main():
    # setting the app title
    st.title("Movie Success Predictor")
    # creating input widgets for user input
    rating = st.selectbox("Rating" , ['adults', 'adults guidance', 'all audiance', 'not rated'])
    genre = st.selectbox("Genre" , ['Drama', 'Adventure', 'Action', 'Comedy', 'Horror', 'Biography',
       'Crime', 'Other', 'Animation'])
    votes = st.slider("Votes" , min_value= 0.0 , max_value=2400000.0 , value=0.0 ,step=100.0)
    year = st.slider("Year" , min_value= 1980 , max_value=2020 , value=0 ,step=1)
    country = st.selectbox("Country" ,['United Kingdom', 'United States', 'Other', 'Canada', 'France',
       'Germany'] )
    rest_type_counts = st.selectbox("Number of Restaurant Type " , [1,2])
    budget = st.slider( "Budget" , min_value = 3000.0 , max_value = 356000000.0 , value = 0.0 , step = 1000.0)
    gross = st.slider( "Gross" , min_value = 309.0 , max_value = 356000000.0 , value = 0.0 , step = 1000.0)
    runtime = st.slider( "Runtime" , min_value = 63.0 , max_value = 366.0 , value = 0.0 , step = 5.0)
    # predict button
    if st.button("Predict"):
        # calling the prediction function with user inputs
        results = prediction(rating, genre, year, votes, country, budget, gross, runtime)
        label = ["Unsuccessful" , "Successful"]
        st.text(f"The Movie will be {label[results]}.")
        
if __name__ == '__main__':
    main()    
    
