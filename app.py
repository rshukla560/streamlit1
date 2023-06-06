from sklearn.ensemble import RandomForestClassifier

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# converting csv with feature and label to df
df = pd.read_csv("data.csv")

target = "target"
X = df.drop(columns = target)
y = df[target]


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)

#intiailizing random forest with default parameters
model = RandomForestClassifier(random_state=42)
# traingi the model
model.fit(X_train, y_train)

filename = 'model_flower.pickle'
#saving the model for deploying
joblib.dump(model, filename)
