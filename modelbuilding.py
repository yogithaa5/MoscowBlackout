import pandas as pd
Nimda = pd.read_csv('Nimda_cleaned.csv')

df = Nimda.copy()
target = 'Labels'

target_mapper = {'Normal':0, 'Virus':1,}
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
pickle.dump(clf, open('Nimda_clf.pkl', 'wb'))