import pandas as pd
df = pd.read_csv(r'C:\Users\aaron\Documents\heart.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model_rf = RandomForestClassifier()

model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)


import pickle
filename = 'model.pkl'
pickle.dump(model_rf, open(filename, 'wb'))

