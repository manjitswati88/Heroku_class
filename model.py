#importing the libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

dataset=pd.read_csv("/content/hiring.csv")

dataset['experience'].fillna(0,inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

X=dataset.iloc[:,:3]

#converting word into integer value
def convert_to_int(word):
  word_dict = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,0:0}
  return word_dict[word]

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

Y=dataset.iloc[:,-1]
#splitting training and test set
#since we have  a very small dataset,we will trains

from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=5)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with training data  
regressor.fit(X, Y)

#saving model to disk
pickle.dump(regressor,open("model.pkl","wb"))


#loading model to compare the results
model = pickle.load(open("model.pkl","rb"))
print(model.predict([[2,9,6]]))

