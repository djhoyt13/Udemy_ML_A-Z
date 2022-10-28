import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import streamlit as st

wine_data = pd.read_csv(r'winequality-red.csv')

st.title("Red Wine Quality Prediction")
if st.checkbox('Show dataframe'):
    st.dataframe(wine_data)

X = wine_data.iloc[:, :-1].values
y = wine_data.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# create random input
def rand_input(df, field):
    upper = df[field].max()
    lower = df[field].min()
    return random.uniform(lower,upper)

# creates field of random inputs

def rand_field(df):
    rand_lst = []
    y_col = str((list(df.iloc[0:0,-1:]))[0])
    X_df = df.drop(y_col, axis=1)
    for i in X_df:
        rand_lst.append(rand_input(X_df, i))
    return rand_lst

# Creates empty df for randomized X variables


# Adds newly generated data to random_var_df
if st.button('Create Random Input'):
    random_var_df = pd.DataFrame(columns=list(wine_data.columns)[:-1])
    random_var_df.loc[len(random_var_df)] = rand_field(wine_data)
    st.dataframe(random_var_df)
else:
    st.write('Click the Button Dummy')