from flask import Flask
from flask import request
import pandas as pd
import joblib


app = Flask(__name__)

@app.route('/predice', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_, index=[0])
    query = pd.get_dummies(query_df)

    classifier = joblib.load('models/modelo_prueba_1.joblib')
    prediction = classifier.predict(query)

    if prediction[0] :
        return "TRUE: El paciente no tiene diabetes"
    else :
        return "FALSE: El paciente si tiene diabetes"


if __name__ == "__main__":
    app.run(port=8000, debug=True)