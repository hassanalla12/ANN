# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:33:35 2022

@author: IHICHR
"""

#1-prepration des données 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#Geography column
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])],remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[: , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                                                   

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#2 construire le reseau de neurone ANN

#importation des modules keras 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Etape 1 : Initiation du ANN

classifier = Sequential()

#Ajouter la couche d'entrée et une couche cachée
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform",input_dim=11))

classifier.add(Dropout(rate=0.1))
#Ajouter une deuxieme couche cachée 
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.1))

#Ajouter une couche de sortie 
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

#Compilation

classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics=["accuracy"])

#Entrainement du reseau 

classifier.fit(X_train,y_train, batch_size=10, epochs=100)


#prediction the test 

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

#♣Confusion matrix 
 from sklearn.metrics import confusion_matrix

 cm = confusion_matrix(y_test , y_pred)

"""Pays : Espagne
Score de crédit : 450
Genre : Feminin
Âge : 30 ans
Durée depuis entrée dans la banque : 8 ans
Balance : 5000 
Nombre de produits : 1
Carte de crédit ? Oui
Membre actif ? : non
Salaire estimé : 25000 €"""

#Solution : 

client_test = np.array([[0.0 ,1 , 450 , 1 ,30 ,8 ,5000 , 1, 1 , 0, 25000]])
my_predicition = classifier.predict(sc.transform(client_test))
print(my_predicition)
my_prediction = (my_predicition > 0.5)
print(my_prediction)

#Evaluation de notre Modele 

from  keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def build_classifier() : 
    
    classifier = Sequential()

    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform",input_dim=11))
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
           
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
        
    classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

precision = cross_val_score(estimator= classifier, X=X, y=y , cv= 11 )

moyenne = precision.mean()
ecart_type = precision.std()

print(precision)
print(moyenne)
print(ecart_type)

#partie amelioration ANN

from  keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer) : 
    
    classifier = Sequential()
    
    #Ajouter la couche d'entrée et une couche cachée
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform",input_dim=11))
    #Ajouter une deuxieme couche cachée 
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    #Ajouter une couche de sortie 
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
   #Compilation
    classifier.compile(optimizer= optimizer, loss= "binary_crossentropy", metrics=["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size" : [ 25, 32],
              "epochs" : [100 , 500] ,
              "optimizer" : ["adam","rmsprop"]}

gridsearch =GridSearchCV(estimator = classifier , param_grid= parameters , scoring="accuracy",cv=10)

gridsearch = gridsearch.fit(X_train, y_train)

best_parametrs = gridsearch.best_parametrs
best_precision = gridsearch.best_precision







































































