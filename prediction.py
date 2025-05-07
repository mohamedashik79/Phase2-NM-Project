import pickle

def predict_severity(input_data, model):
    prediction = model.predict([input_data])
    return prediction
