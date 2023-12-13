import pandas as pd
import altair as alt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression




## ***********************************************

s = pd.read_csv ("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)  
    return x


ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'web1h']].copy()
 
# Handle the 'income' column as ordered numeric from 1 to 9, above 9 considered missing
ss['income'] = np.where((ss['income'] >= 1) & (ss['income'] <= 9), ss['income'], np.nan)
 
# Handle the 'education' column as ordered numeric from 1 to 8, above 8 considered missing
ss['educ2'] = np.where((ss['educ2'] >= 1) & (ss['educ2'] <= 8), ss['educ2'], np.nan)
 
# Apply clean_sm to the 'par' column
ss['par'] = np.where(ss['par'] == 1, 0, 1)
 
# Apply clean function to marital
ss['marital'] = clean_sm(s['marital'])
 
# Apply clean_sm to the 'gender' column
ss['gender'] = clean_sm(ss['gender'])
 
# Handle the 'age' column as numeric, above 98 considered missing
ss['age'] = np.where(ss['age'] <= 98, ss['age'], np.nan)
 
# Apply clean_sm to the target column
ss['sm_li'] = ss['web1h'].apply(clean_sm)
 
# Drop missing values
ss = ss.dropna()


# Separate the target column 'sm_li' from the DataFrame
y = ss['sm_li']  # Target vector

# Selecting the features (excluding the target column 'sm_li')
X = ss.drop(['sm_li', 'web1h'], axis=1)  # Feature set


# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
#scaled_features = scaler.transform(features)

#scaler = StandardScaler()
#X = scaler.fit_transform(X)


# Logistic regression model 
logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# Fit model
logistic_model.fit(X_train, y_train)


# Use the trained model to make predictions on the test set
y_pred = logistic_model.predict(X_test)


# STREAMLIT


st.title("She's a LinkedIn User!")
st.title("He's a LinkedIn User!")
st.title('Are :red[YOU] a LinkedIn User?')

st.subheader('Created By: Landon Jones')

st.divider()
st.write("LinkedIn's staggering platform expansion and surge in globally engaged users in the past five years warrants admiration.")  

st.write("This popular social media platform operates via websites and mobile apps that allow members to create professional profiles and connect with each other for networking or career opportunities.  Users can establish connections, join groups, follow companies, and share content as a way to build their professional brand and find career options LinkedIn's staggering platform expansion and surge in globally engaged users in the past five years warrants admiration.")  

st.write("My application performs a complex analysis of current Pew Research social media usage data to power an easy-to-use tool that makes instant predictions regarding a person's use of LinkedIn in particularly by conducting an in-depth examination of current Pew Research usage statistics. ")

st.write("Complete the 5 questions below and....")
st.divider()


# age slider
age = st.slider('Kindly specify your age?', 18, 98)
st.write("I'm ", age, 'years old')

# Horizontal radio buttons
radio_col1, radio_col2, radio_col3 = st.columns(3)

with radio_col1:
   female = st.radio('Female', ('Yes', 'No'))

with radio_col2: 
   parental_status = st.radio("Parent?", ('Yes','No'))
   
with radio_col3:
   marriage_status = st.radio("Married?", ('Yes','No'))



# Define a dictionary that maps numbers to income ranges
income_options = {
    1: "1-Less than $10,000",
    2: "2-10 to under $20,000",
    3: "3-20 to under $30,000",
    4: "4-30 to under $40,000",
    5: "5-40 to under $50,000",
    6: "6-50 to under $75,000",
    7: "7-75 to under $100,000",
    8: "8-100 to under $150,000",
    9: "9-$150,000 or more"
}
 
# Collect user input
income_range = st.selectbox('Income Range', options=list(income_options.values()))
 
# Extract the number from the selected income range
income_range_encoded = int(income_range.split('-')[0])
 
# Define a dictionary that maps numbers to education levels
education_options = {
    1: "1-Less than high school (Grades 1-8 or no formal schooling)",
    2: "2-High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "3-High school graduate (Grade 12 with diploma or GED certificate)",
    4: "4-Some college, no degree (includes some community college)",
    5: "5-Two-year associate degree from a college or university",
    6: "6-Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "7-Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    8: "8-Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
}
 
# Collect user input
education_level = st.selectbox('Education Level', options=list(education_options.values()))
 
# Extract the number from the selected education level
education_level_encoded = int(education_level.split('-')[0])
 

# Binary Encoding
parental_status_encoded = 1 if parental_status =='Yes' else 0
marriage_status_encoded = 1 if marriage_status =='Yes' else 0
female_encoded = 1 if female == 'Yes' else 0

# Creating a feature vector
features = [[income_range_encoded, education_level_encoded, parental_status_encoded, marriage_status_encoded, female_encoded, age]]
 





 
# Centered button for prediction
if st.button('Prediction'):
    # Make prediction
    prediction = logistic_model.predict(features)

    # Get predicted probabilities
    proba = logistic_model.predict_proba(features)[0]

    # Show prediction  
    if prediction[0] == 1:
        st.write(f'The model predicts the person is likely to use LinkedIn, with a probability of {proba[1]:.2%}')
    else:
        st.write(f'The model predicts the person is unlikely to use LinkedIn, with a probability of {proba[0]:.2%}')





