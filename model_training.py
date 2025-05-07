from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_clean_data
import pandas as pd

df = load_and_clean_data('../data/sample_accident_data.csv')
X = df.drop('Severity', axis=1)
y = df['Severity']

# Dummy encoding for simplicity
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully.")
