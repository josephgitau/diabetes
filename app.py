# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# App title
st.title("Diabetes Detection System")

# heading 
st.header("This is a diabetes detection system")
st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning. ")

# load data
df = pd.read_csv("diabetes.csv")

# Display sample data to the user
st.subheader("Sample Data")
st.dataframe(df.head())

# Data Preprocessing

# List of columns to update
columns_to_update = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN in the specified columns
df[columns_to_update] = df[columns_to_update].replace(0, np.nan)

# Replacing missing values
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp

# Glucose
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3

# Blood pressure
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3

# Skin thickness
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0

# Insulin
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8

# BMI
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4

# Add two columns for data visualization
col1, col2 = st.columns(2)

with col1:
    st.header("Correlation Heatmap")
    f, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f)   

with col2:
    # Display a clustermap
    st.header("Clustermap")
    f, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f) 

# Create our data features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Select columns to fit the model
columns = ['Insulin','Glucose','BMI','Age','SkinThickness']
X = X[columns]

# Scale data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# get user input
name = st.text_input('What is your name?').capitalize()

# get feature input from user
if name != "":
    st.write("Hello {} Please complete the form below".format(name))
else:
    st.write("Please enter your name")

# Get user input
def get_user_input():
    insulin = st.sidebar.slider("Insulin", 0.0, 846.0, 30.5)
    glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    age = st.sidebar.slider("Age", 21, 81, 29)
    skin_thickness = st.sidebar.slider("SkinThickness", 0, 99, 23)

    # Store a dictionary into a variable
    user_data = {
        'Insulin': insulin,
        'Glucose': glucose,
        'BMI': bmi,
        'Age': age,
        'SkinThickness': skin_thickness
    }

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# return user input
user_input = get_user_input()

# Display the user input
st.subheader("Below is the user input {}".format(name))
st.dataframe(user_input)

# scale the user input
user_input_scaled = scaler.transform(user_input)
user_input_scaled = pd.DataFrame(user_input_scaled, columns=columns)

# Create a button to ask user to get result
bt = st.button("Get Result")

if bt:
    # Create a Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Get the user input features prediction
    prediction = model.predict(user_input_scaled)

    # Display the result
    if prediction == 1:
        st.write("Hello {}, you have diabetes".format(name))
    else:
        st.write("Hello {}, you do not have diabetes".format(name))
    # Diplay model accuracy
    st.write("Model Accuracy: ", round(metrics.accuracy_score(y_test, model.predict(X_test)), 2)*100)





