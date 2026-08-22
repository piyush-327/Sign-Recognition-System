import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('dataset.csv')

y = df['Label']
X = df.drop(columns=['Label'])

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

clf = RandomForestClassifier()
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

with open('model1.pkl', 'wb') as f:
    pickle.dump(clf,f)

