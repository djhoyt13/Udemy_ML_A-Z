import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import streamlit as st

wine_data = pd.read_csv(r'winequality-red.csv')

st.title("Red Wine Quality Prediction")
st.subheader("Predict your red wine's quality score from randomly generated attributes")

# Breaks data into dependent and independent variables
X = wine_data.iloc[:, :-1].values
y = wine_data.iloc[:, -1].values

# Trains the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# create random input function
def rand_input(df, field):
    upper = df[field].max()
    lower = df[field].min()
    return random.uniform(lower,upper)

# creates field of random inputs function
def rand_field(df):
    rand_lst = []
    y_col = str((list(df.iloc[0:0,-1:]))[0])
    X_df = df.drop(y_col, axis=1)
    for i in X_df:
        rand_lst.append(rand_input(X_df, i))
    return rand_lst

# Adds newly generated data to random_var_df
if st.button("Create Your Wine's Random Attributes"):
    random_var_df = pd.DataFrame(columns=list(wine_data.columns)[:-1])
    random_var_df.loc[len(random_var_df)] = rand_field(wine_data)
    st.session_state["random_data"] = random_var_df

# Freezes user generated input data table
if "random_data" in st.session_state:
    st.dataframe(st.session_state['random_data'])

# Predicts quality score based on newly generated input
if st.button("Get Your Wine's Quality Score"):
    new_lst = st.session_state.random_data.iloc[(len(st.session_state.random_data.index)-1), :].values.tolist()  # most recent observation to list
    new_array = np.array(new_lst).reshape(1, -1)   # casts to a 2d np.array for prediction input
    prediction = float(regressor.predict(new_array))  # predicts quality score 
    st.metric(label="Your Wine's Quality Score is:", value=prediction)

# Allows users to view the original training data
with st.sidebar:
    if st.checkbox('Show Training Data'):
        st.dataframe(wine_data)