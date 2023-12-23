# Lab12: Classification des fleurs iris en utilisant Scikit_learn
# Réalisé par : SALAH-EDDINE ZIDANI / EMSI LES ORANGERS 2023/2024


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

# Step 1: DataSet
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
print(iris.feature_names)
print(iris.data.shape)


# Step 2: Model
model = RandomForestClassifier()


# Step 3: Train
model.fit(iris.data, iris.target)


# Step 4: Test
prediction = model.predict([[0.9, 1.0, 1.1, 1.8]])
print(prediction)
print(iris.target_names[prediction])
# Web Deployment of the Model: streamlit run filename.py
st.header('iris flowers classification')
st.image("images/iris.png")
st.sidebar.header('iris features')
def user_input():
    sepal_length = st.sidebar.slider('sepal length', 0.1, 9.9, 5.0)
    sepal_width = st.sidebar.slider('sepal width', 0.1, 9.9, 5.0)
    petal_length = st.sidebar.slider('petal length', 0.1, 9.9, 5.0)
    petal_width = st.sidebar.slider('petal width', 0.1, 9.9, 5.0)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features

df = user_input()
st.write(df)
selected_model = st.sidebar.selectbox('Select your learning algorithm', ['RandomForest', 'DecisionTree', 'KRR', 'SVM'])
st.write('Your selected algorithm is : ', selected_model)
st.subheader('prediction')
prediction = model.predict(df)
st.write(prediction)
st.write(iris.target_names[prediction])
st.image("images/"+iris.target_names[prediction][0]+".png")
