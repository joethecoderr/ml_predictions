import joblib
import numpy as np

from flask import Flask
from flask import jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app,supports_credentials=True, resources={r'/.*': {'origins': '*'}})
app.config['CORS_HEADERS'] = 'Content-Type'
#cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/happiness', methods=['GET'])
def predict_happiness():
    model = joblib.load('./models/best_model.pkl')
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'Happiness score prediction' : list(prediction)})

@app.route('/heart', methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type' ], supports_credentials=True)
def predict_heart_disease():
    if request.method == 'POST':
        age = request.form.get('age')
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        restbps = request.form.get('restbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')
        heart_disease = 'No'
        model = joblib.load('./models/best_model_score_0.9678_heartGradientClass.pkl')
        X_test = np.array([int(age),int(sex),int(cp),int(restbps),int(chol),int(fbs),int(restecg),int(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal)])
        prediction = model.predict(X_test.reshape(1,-1))
        if prediction[0] == 1:
            heart_disease = 'Yes'
            return jsonify({'Heart disease' : 1})
        else:
            return jsonify({'Heart disease' : 0})
        
if __name__ == "__main__":
    app.run()