from flask import Flask, render_template, request
from jinja2 import TemplateNotFound
import numpy as np
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('Model.pkl', 'rb'))

@app.route('/', defaults = {'page': 'Home.html'})
@app.route('/<page>')
def html_lookup(page):
    return render_template('{}'.format(page))

#To use the predict button in our web-app
@app.route('/YieldPrediction',methods = ['POST'])
def YieldPrediction():
    
    if request.method == "POST":
       season = request.form.get("season")
       area = request.form.get("area")
       temperature = request.form.get("temperature")
       pH = request.form.get("pH")
       rainfall = request.form.get("rainfall")
       phosphorous = request.form.get("phosphorous")
       nitrogen = request.form.get("nitrogen")
       potassium = request.form.get("potassium")
       crop = request.form.get("crop")
    
    prediction = model.predict([[season, area, temperature, pH, rainfall, phosphorous, nitrogen, potassium, crop]])

    return render_template('YieldPrediction.html', prediction_text = 'The crop yield is: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
