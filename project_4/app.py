from flask import Flask, render_template, request
import numpy as np
import pickle


model_km = pickle.load(open("model/model_km.pkl", "rb"))

scaler = pickle.load(open('model/model_scaler.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    BALANCE = float(request.form["BALANCE"])
    PURCHASES = float(request.form["PURCHASES"])
    CASH_ADVANCE = float(request.form["CASH_ADVANCE"])
    CREDIT_LIMIT = float(request.form["CREDIT_LIMIT"])
    PAYMENTS = float(request.form["PAYMENTS"])
    MINIMUM_PAYMENTS = float(request.form["MINIMUM_PAYMENTS"])
   
    feature = [BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS]

    feature = scaler.fit_transform([feature])
    prediction = model_km.predict(feature)

    return render_template("index.html", prediction_text="Cluster Anda adalah : {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
