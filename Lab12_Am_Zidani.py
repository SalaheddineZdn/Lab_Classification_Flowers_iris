# Amelioration lab12: Classification of flowers iris using Scikit_learn
# Réalisé par : SALAH-EDDINE ZIDANI / EMSI LES ORANGERS 2023/2024

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import streamlit as st
import pandas as pd

# Step 1: DataSet
iris = datasets.load_iris()

# Step 2: Model
def train_model(selected_model):
    if selected_model == 'RandomForest':
        model = RandomForestClassifier()
    elif selected_model == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif selected_model == 'KRR':
        model = KernelRidge()
    elif selected_model == 'SVM':
        model = SVC()
    elif selected_model == 'LogisticRegression':
        model = LogisticRegression()
    elif selected_model == 'KNeighbors':
        model = KNeighborsClassifier()
    elif selected_model == 'NaiveBayes':
        model = GaussianNB()
    elif selected_model == 'NeuralNetwork':
        model = MLPClassifier()
    elif selected_model == 'GradientBoosting':
        model = GradientBoostingClassifier()
    elif selected_model == 'AdaBoost':
        model = AdaBoostClassifier()
    elif selected_model == 'Bagging':
        model = BaggingClassifier()
    else:
        model = None
    return model

# Web Deployment of the Model: streamlit run Lab12_Am_Zidani.py
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

selected_model = st.sidebar.selectbox('Select your learning algorithm',
                                      ['RandomForest', 'DecisionTree', 'KRR', 'SVM',
                                       'LogisticRegression', 'KNeighbors', 'NaiveBayes',
                                       'NeuralNetwork', 'GradientBoosting', 'AdaBoost', 'Bagging'])
st.write('Your selected algorithm is : ', selected_model)

# Step 3: Train Model
model = train_model(selected_model)
model.fit(iris.data, iris.target)

# Step 4: Test and Make Predictions
st.subheader('Prediction')
prediction = model.predict(df)
st.write(prediction)
st.write(iris.target_names[prediction])
st.image("images/" + iris.target_names[prediction][0] + ".png")