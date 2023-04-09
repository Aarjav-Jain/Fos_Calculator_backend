from flask import Flask,request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# we define a variable called app
app = Flask(__name__) 

modelML = load_model('model_NN_accurate.h5')
modelSC = load_model('model_safety_chart.h5')

# TODO: 
# make two routes
# load the model 
# extract values from request body preably as an array
# convert extracted values to number for prediction
# return the predicted value

def makePredictionML(data):
    newData = np.array(data)
    prediction = modelML.predict(newData.reshape(1,11))
    return(prediction[0][0])

def makePredictionSafetyChart(data):
    newData = np.array(data)
    prediction = modelSC.predict(newData.reshape(1,3))
    return (prediction[0][0])

@app.route("/models/machine_learning",methods=['POST'])
def MLResult():
    if(request.method == 'POST'):
        data = request.get_json(force=True)
        predictedValue = makePredictionML(data)
        return str(predictedValue)
    
@app.route("/models/safety_chart",methods=['POST'])
def SafetyChartResult():
    if(request.method == 'POST'):
        data = request.get_json(force=True)
        # print(data)
        predictedValue = makePredictionSafetyChart(data)
        return str(predictedValue)
        # return 'data recieved'

if __name__ == '__main__':
    app.run(debug=True)