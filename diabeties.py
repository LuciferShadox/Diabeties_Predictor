#importing library finctions
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #split dataset into test and train cases
from sklearn.naive_bayes import GaussianNB # import library of NB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def get_data(filename):
    dataset=pd.read_csv(filename)
    x=dataset.iloc[:,0:8]#select all attribute columns
    y=dataset.iloc[:,8]#select outputs
    return x,y


# train data
x,y=get_data('diabetes.csv')
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0, test_size=0.20)
classifier=GaussianNB()#using naivce bayes model
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test) #predict output



cm=confusion_matrix(y_test,y_pred)#comparing with test cases
print "confusion matrix"
print cm
print "Accuracy:"
print(accuracy_score(y_test,y_pred))

test_output=[['5','116','74','0','0','25.6','0.201','30']]#test output
test_output=np.array(test_output).reshape(1,-1)
test_output=test_output.astype(np.float64)
pred=classifier.predict(test_output)
print pred
if (pred==1):
    print "has diabeties"
else :
    print "no diabeties"






    
        

