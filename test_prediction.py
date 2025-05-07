import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_clean_data

# Load and prepare data
df = load_and_clean_data('sample_accident_data.csv')
X = pd.get_dummies(df.drop('Severity', axis=1))
y = df['Severity']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Use the first row as sample input
sample_input = X.iloc[0].values
prediction = model.predict([sample_input])
print("Predicted Severity:", prediction[0])
