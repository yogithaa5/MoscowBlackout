import pandas as pd
moscow = pd.read_csv('moscow_cleaned.csv')

df = moscow.copy()
target = 'Labels'

target_mapper = {'Normal':0, 'moscow':1,}
def target_encode(val):
    return target_mapper[val]

df[target] = df[target].apply(target_encode)

# Separating X and Y
X = df.drop(target, axis=1)
Y = df[target]

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('moscow_clf.pkl', 'wb'))