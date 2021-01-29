import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/happiness', methods=['GET'])
def predict_happiness():
    model = joblib.load('./models/best_model.pkl')
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'Happiness score prediction' : list(prediction)})

@app.route('/heart', methods=['GET'])
def predict_heart_disease():
    heart_disease = 'No'
    model = joblib.load('./models/best_model_score_0.9678_heartGradientClass.pkl')
    X_test = np.array([71,0,0,112,149,0,1,125,0,1.6,1,0,2])
    prediction = model.predict(X_test.reshape(1,-1))
    if prediction[0] == 1:
        heart_disease = 'Yes'
    return 'Have a heart disease?: ' + heart_disease
if __name__ == "__main__":
    app.run()