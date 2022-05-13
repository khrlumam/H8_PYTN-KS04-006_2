from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

# Model Lyft
model_lyft = pickle.load(open("model/lyft_model.pkl", "rb"))
# Model Uber
model_uber = pickle.load(open("model/uber_model.pkl", "rb"))


app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


# Lyft
@app.route('/predict_1', methods=['POST'])
def predict_1():
    '''
    For rendering results on HTML GUI
    '''
    features_1 = [x for x in request.form.values()]
    final_features_1 = [np.array(features_1)]
    prediction_1 = model_lyft.predict(final_features_1)

    output_1 = np.round(prediction_1[0], 2)

    return render_template('index.html', prediction_text_1='Prediksi Tarif Lyft : $ {}'.format(output_1))

# Uber
@app.route('/predict_2', methods=['POST'])
def predict_2():
    '''
    For rendering results on HTML GUI
    '''
    features_2 = [y for y in request.form.values()]
    final_features_2 = [np.array(features_2)]
    prediction_2 = model_uber.predict(final_features_2)

    output_2 = np.round(prediction_2[0], 2)

    return render_template('index.html', prediction_text_2='Prediksi Tarif Uber : $ {}'.format(output_2))

if __name__ == '__main__':
    app.run(debug=True)
