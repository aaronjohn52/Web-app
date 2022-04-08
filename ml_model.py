import pandas as pd
df = pd.read_csv(r'C:\Users\aaron\Documents\heart.csv')

import lightgbm as lgb
from lightgbm import LGBMClassifier


from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model_lgbm = LGBMClassifier()
model_lgbm.fit(X_train, y_train)

y_pred_lgbm = model_lgbm.predict(X_test)

import pickle
filename = 'model.pkl'
pickle.dump(model_lgbm, open(filename, 'wb'))

