import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

#  LOAD DATA 
df = pd.read_csv("gesture_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

#  SPLIT 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  TRAIN MODEL 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#  EVALUATE 
y_pred = clf.predict(X_test)
print("\n   Classification Report   ")
print(classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))

#  SAVE MODEL 
joblib.dump(clf, "gesture_model.pkl")
print(" Model saved as gesture_model.pkl")
