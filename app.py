from flask import Flask, render_template, request
from jinja2 import TemplateNotFound
import numpy as np
import pickle
from werkzeug.utils import secure_filename
from flask import jsonify
import joblib
import csv
import requests
import json
from Server.Models.crop_class import *
from Server.Models.production_class import *
from Server.Models.crop_disease_dict import crop_dict
import tensorflow as tf
import cv2
import sys


#Initialize the flask App
app = Flask(__name__)
YieldPredictionModel = pickle.load(open('Server/Models/YieldPredictionModel.pkl', 'rb'))
FertilizerPredictionModel = pickle.load(open('Server/Models/FertilizerPredictionModel.pkl', 'rb'))
CropRecommendationModel = pickle.load(open('Server/Models/CropRecommendationModel.pkl', 'rb'))

@app.route('/', defaults = {'page': 'Home.html'})
@app.route('/<page>')
def html_lookup(page):
    return render_template('{}'.format(page))


#Crop Yield Prediction
@app.route('/YieldPrediction', methods = ['POST'])
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
    
    prediction = YieldPredictionModel.predict([[season, area, temperature, pH, rainfall, phosphorous, nitrogen, potassium, crop]])

    return render_template('YieldPrediction.html', prediction_text = 'The crop yield is: {}'.format(prediction))

#Fertilizer Prediction
@app.route('/FertilizerPrediction', methods = ['POST'])
def FertilizerPrediction():
    
    if request.method == "POST":
       temperature = request.form.get("temperature")
       humidity = request.form.get("humidity")
       moisture = request.form.get("moisture")
       soil_type = request.form.get("soil_type")
       crop_type = request.form.get("crop_type")
       nitrogen = request.form.get("nitrogen")
       potassium = request.form.get("potassium")
       phosphorous = request.form.get("phosphorous")
    
    prediction = FertilizerPredictionModel.predict([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])

    if prediction == 0:
        fertilizer = "10-26-26"
    elif prediction ==1:
        fertilizer = "14-35-14"
    elif prediction == 2:
        fertilizer = "17-17-17"
    elif prediction == 3:
        fertilizer = "20-20"
    elif prediction == 4:
        fertilizer = "28-28"
    elif prediction == 5:
        fertilizer = "DAP"
    else:
        fertilizer = "Urea"

    return render_template('FertilizerPrediction.html', prediction_text = 'The fertilizer to be used is: {}'.format(fertilizer))

#Crop Recommendation
@app.route('/CropRecommendation', methods = ['POST'])
def CropRecommendation():
    
    if request.method == "POST":
       pH = request.form.get("pH")
       nitrogen = request.form.get("nitrogen")
       phosphorous = request.form.get("phosphorous")
       potassium = request.form.get("potassium")
       oc = request.form.get("oc")
       particles = request.form.get("particles")
       waterholdingcontent = request.form.get("waterholdingcontent")
       soil_type = request.form.get("soil_type")
    
    prediction = CropRecommendationModel.predict([[pH, nitrogen, phosphorous, potassium, oc, particles, waterholdingcontent, soil_type]])
    
    if prediction[0] == 1:
        crop = "Carrot"
    elif prediction[0] == 2:
        crop = "Coconut"
    elif prediction[0] == 3:
        crop = "Cotton"
    elif prediction[0] == 4:
        crop = "Groundnut"
    elif prediction[0] == 5:
        crop = "Melon"
    elif prediction[0] ==6:
        crop = "Millet"
    elif prediction[0] == 7:
        crop = "Potato"
    elif prediction[0] == 8:
        crop = "Rice"
    elif prediction[0] == 9:
        crop = "Vegetable"
    elif prediction[0] == 10:
        crop = "Wheat"

    return render_template('CropRecommendation.html', prediction_text = 'The recommended crop is: {}'.format(crop))

'''
#########crop recommendation #########

scale_val = 0.1

month_dict = {
    'JAN': 0,
    'FEB': 1,
    'MAR': 2,
    'APR': 3,
    'MAY': 4,
    'JUN': 5,
    'JUL': 6,
    'AUG': 7,
    'SEP': 8,
    'OCT': 9,
    'NOV': 10,
    'DEC': 11
}


def get_avg(temps, predict_month):
    temp_arr = []
    idx_num = month_dict[predict_month]
    temp_arr.append(float(temps[idx_num]))
    for i in range (0, 5, 1):
        idx_num += 1
        idx_num = idx_num % 12
        temp_arr.append(float(temps[idx_num]))
    return np.average(temp_arr, axis=0)

def get_ground_water(ground_water, predict_month, district):
    temp_arr=[]
    gwater = list(ground_water[district].values())
    idx_num = month_dict[predict_month]
    temp_arr.append(gwater[idx_num])
    for i in range(0, 5, 1):
        idx_num += 1
        idx_num = idx_num % 12
        temp_arr.append(gwater[idx_num])
    print("gwater:  ", temp_arr)
    return np.average(temp_arr, axis=0)

def get_soil(soil_type):
    soil_arr=[]
    if(soil_type == 'Alluvial'):
        soil_arr.append(1)
    else:
        soil_arr.append(0)

    if(soil_type == 'Black'):
        soil_arr.append(1)
    else:
        soil_arr.append(0)

    if(soil_type == 'Loam'):
        soil_arr.append(1)
    else:
        soil_arr.append(0)

    if(soil_type == 'Red'):
        soil_arr.append(1)
    else:
        soil_arr.append(0)

    return soil_arr
        

# AI/ML parameteres default
nn_weight_path = 'Server/Models/weights/kharif_crops_final.pth'
production_weight_path = 'Server/Models/weights/production_weights.sav'


@app.route('/CropRecommendation',methods=['post'])
def CropRecommendation():
    area = request.form.get('area')
    potassium = request.form.get('potassium')
    nitrogen = request.form.get('nitrogen')
    phosphorous = request.form.get('phosphorous')
    ph = request.form.get('ph')
    crop_season = request.form.get('crop_season')
    current_crop = request.form.get('current_planted_crop')
    predict_month = request.form.get('predict_month')
    is_current = request.form.get('is_current')
    soil_type = request.form.get('soil_type')

    # Use this API for finding data using latitudes and Longitudes
    # https://climateknowledgeportal.worldbank.org/api/data/get-download-data/projection/mavg/tas/rcp26/2020_2039/21.1458$cckp$79.0882/21.1458$cckp$79.0882
    # temps stores the predicted temperature
    #latitude = str(request.form.get('lat'))
    #longitude = str(request.form.get('lng'))
    
    pin_code=str(request.form.get('pin_code'))
    district = (request.form.get('district')).upper()
    state = (request.form.get('state')).upper()
    URL="http://api.positionstack.com/v1/forward?access_key=0e76df9e3416fbe7863ea96d1b693b00&query="+pin_code+"%20"+district+"%20"+state
    resp=requests.get(url=URL)

    # latitude = str(request.form.get('lat'))
    # longitude = str(request.form.get('lng'))

    pin_code=request.form.get('pin_code')
    pin_code=str(pin_code)
    district = (request.form.get('district')).upper()
    state = (request.form.get('state')).upper()


    URL="http://api.positionstack.com/v1/forward?access_key=0e76df9e3416fbe7863ea96d1b693b00&query="+pin_code+"%20"+district+"%20"+state
    resp=requests.get(url=URL)

    result_dict=resp.json()['data'][0]
    latitude=result_dict['latitude']
    longitude=result_dict['longitude']


    param = "tas"
    URL = "https://climateknowledgeportal.worldbank.org/api/data/get-download-data/projection/mavg/"+ param +"/rcp26/2020_2039/" + \
        latitude+"$cckp$"+longitude + "/"+latitude + "$cckp$"+longitude + ""
    
    
    resp = requests.get(url=URL)
    decoded = resp.content.decode("utf-8")
    cr = csv.reader(decoded.splitlines(), delimiter=',')
    my_list = list(cr)
    temps = []
    for index, row in enumerate(my_list):
        if index == 0:
            continue
        if index > 13:
            break
        temps.append(row[0])

    param = "pr"
    resp = requests.get(url=URL)
    decoded = resp.content.decode("utf-8")
    cr = csv.reader(decoded.splitlines(), delimiter=',')
    my_list = list(cr)
    rainfall = []
    for index, row in enumerate(my_list):
        if index == 0:
            continue
        if index > 13:
            break
        rainfall.append(row[0])

    # Getting the current temperature (if Current=true in Input)
    current_weather_url = "http://api.openweathermap.org/data/2.5/weather?lat="+latitude +"&lon="+longitude +"&APPID=b9bb7acaa4566f8f7de584f90c2b12c2"
    resp = requests.get(current_weather_url)
    decoded = resp.content.decode("utf-8")
    resp = json.loads(decoded)
    current_temp = resp["main"]["temp"]
    #print(current_temp)


    # Do the prediction here using Classifier clf.
    print(crop_season)
    if(crop_season == 'Kharif'):
        nn_weight_path = 'Server/Models/weights/kharif_crops_final.pth'
    elif(crop_season == 'Rabi'):
        nn_weight_path = 'Server/Models/weights/rabi_crops_final.pth'
    elif(crop_season == 'Zaid'):
        nn_weight_path = 'Server/Models/weights/zaid_crops_final.pth'
    
    production_weight_path = 'Server/Models/weights/production_weights.sav'

    # get avg values
    temp_avg = get_avg(temps, predict_month)
    rain_avg = get_avg(rainfall, predict_month)

    # gwater calculations
    ground_water_avg = get_ground_water(ground_water, predict_month, district)
    max_area_dist = int(max_area[district])
    # print("gwater avg: {}  max_area_dist:  {}  area:  {}".format(type(ground_water_avg), type(max_area_dist), type(area)))
    gwater_available = scale_val * (float(ground_water_avg) * float(area) ) / float(max_area_dist)
    total_water = rain_avg + gwater_available

    # sow_temp
    if(is_current):
        sow_temp = current_temp
    else:
        sow_temp = temps[month_dict[predict_month]]

    # harvest temp
    harvest_temp = temps[(month_dict[predict_month]+5)%12]

    # soil paramteres
    soil = get_soil(soil_type)

    # Create parameter list
    parameteres = [[temp_avg, ph, total_water, sow_temp, harvest_temp, nitrogen, potassium, phosphorous, soil[0], soil[1], soil[2], soil[3]]]

    # create model instance
    nn_model = crop_model(crop_season)
    #load weights
    nn_model.load_weights(nn_weight_path)
    #get predictions
    pred = nn_model.get_predictions(parameteres)
    pred = pred.detach().numpy()

    # get top_3 predictions
    nn_model.get_top_n_predictions(pred, 3)

    print(nn_model.max_pred_array)

    # Crop Price Prediction
    crop = [str(nn_model.max_pred_array[0][1]), str(nn_model.max_pred_array[1][1]), str(nn_model.max_pred_array[2][1])]
    price_model = Production(crop, int(area), production_weight_path)

    #calculate the production and price and also display
    price_model.calculate_production_price() 

    print(price_model.prod_arr)



    # Making the response message
    response = {
        "predict": [
            {
                "crop": nn_model.max_pred_array[0][1],
                "yield_percent": nn_model.max_pred_array[0][0],
                "production": price_model.prod_arr[0][0],
                "price": price_model.prod_arr[0][1]
            },
            {
                "crop": nn_model.max_pred_array[1][1],
                "yield_percent": nn_model.max_pred_array[1][0],
                "production": price_model.prod_arr[1][0],
                "price": price_model.prod_arr[1][1]
            },
            {
                "crop": nn_model.max_pred_array[2][1],
                "yield_percent": nn_model.max_pred_array[2][0],
                "production": price_model.prod_arr[2][0],
                "price": price_model.prod_arr[2][1]
            }
        ]
    }

    return "crop_recommendation"



'''

####disease detection

model=tf.keras.models.load_model("Server/Models/disease_predictor")

@app.route('/DiseaseDetection',methods = ['POST'])
def DiseaseDetection():
    if request.method == "POST":
        file = request.files['image'].read()

        #convert string data to numpy array
        npimg = np.fromstring(file, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        img=cv2.resize(img,(150,150))
        img=np.array([img])/255.
        predictions = model.predict(img)
        result=predictions.tolist()[0]
        position=result.index(max(result))
        disease=crop_dict(position) 
        
        
    return render_template('DiseaseDetection.html', prediction_text = 'The disease is found out to be: {}'.format(disease)) 



'''
    
if __name__ == "__main__":
    f = open('dataset/ground_water_dic.pkl','rb')
    ground_water = pickle.load(f)
    f.close()
    
    f = open('/max_area_groundwater.pkl','rb')
    max_area = pickle.load(f)
    f.close()
    
    app.run(debug=True)'''

if __name__ == "__main__":
    app.run()
