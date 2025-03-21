# This app is used to develop the AI powered housing price predictors on Streamlit
import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras
import tensorflow as tf
# === Load pre-trained models and preprocessors ===
# (Ensure that the following files are in your repository)
# model_selected = keras.models.load_model('model_selected.h5',
custom_objects={'mse': tf.keras.losses.mean_squared_error}
# model_all = keras.models.load_model('model_all.h5', custom_objects={'mse':
tf.keras.losses.mean_squared_error)
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')
# === Define default values for non-essential features for the full model ===
# This dictionary should include every feature expected by model_all.
# Here we update the defaults for the essential features based on user input later.
default_values = {
'Age': 30, # Will be overwritten by user input
'Gr Liv Area': 1500, # Will be overwritten by user input
'Lot Area': 8000, # Will be overwritten by user input
'Overall Qual': 6, # Will be overwritten by user input
'Neighborhood': 'CollgCr', # Will be overwritten by user input
# Add other non-essential features with default values:
# 'Year Built': 1970,
# 'Mas Vnr Area': 100,
# 'Overall Cond': 5,
# ... (include all additional features as needed)
}
st.title("Ames Housing Price Prediction")
# Sidebar: let the user choose which model to use
model_choice = st.sidebar.radio("Select Model",
("Essential Features Model", "All Features Model"))
st.header("Input Housing Features (Essential Only)")
# User inputs for essential features
age = st.number_input("Age (Yr Sold - Year Built)", min_value=0, max_value=200,
value=30)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=300,
max_value=10000, value=1500)
lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000,
value=8000)
overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10,
value=6)
neighborhood = st.selectbox("Neighborhood",
options=["CollgCr", "Veenker", "Crawfor", "NoRidge",
"Mitchel"])
# When the user clicks the Predict button
if st.button("Predict Sale Price"):
if model_choice == "Essential Features Model":
# Build a DataFrame from the essential features only
input_data = pd.DataFrame({
'Age': [age],
'Gr Liv Area': [gr_liv_area],
'Lot Area': [lot_area],
'Overall Qual': [overall_qual],
'Neighborhood': [neighborhood]
})
# Preprocess input using the selected-features preprocessor
processed_data = preprocessor_selected.transform(input_data)
# Get prediction from the essential-features model
prediction = model_selected.predict(processed_data)
st.success(f"Predicted Sale Price (Essential Model): ${prediction[0]
[0]:,.2f}")
else:
default_all = pd.read_csv('default_all_features.csv', index_col=0)
# Now, 'default_all' contains all the features expected by the
preprocessor.
# Overwrite the essential features with user inputs
default_all.loc[0, 'Age'] = age
default_all.loc[0, 'Gr Liv Area'] = gr_liv_area
default_all.loc[0, 'Lot Area'] = lot_area
default_all.loc[0, 'Overall Qual'] = overall_qual
default_all.loc[0, 'Neighborhood'] = neighborhood
processed_data = preprocessor_all.transform(default_all)
prediction = model_all.predict(processed_data)
st.success(f"Predicted Sale Price (All Features Model): ${prediction[0]
[0]:,.2f}")
