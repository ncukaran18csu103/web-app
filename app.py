from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
import numpy as np
from werkzeug.utils import secure_filename
import os, sys, glob, re



app = Flask(__name__)

#Load pickel file for diabetic
filename = 'diabetes-prediction-rfc-model.pkl'
rfc_diabetic = pickle.load(open(filename, 'rb'))

#Load pickel file for breast cancer
model_bc = pickle.load(open('breast_cancer_lgbm.pkl', 'rb'))

#corona
#file1 = open('xgb_corona.pkl', 'rb')
#clf_corona = pickle.load(file1)

#heart
file2 = open('heart_rfr.pkl', 'rb')
clf_heart = pickle.load(file2)

#liver
model_liver = pickle.load(open('liver_model.pkl', 'rb'))

#pneumonia
#model_pneumonia = load_model(('pneumonia_model.h5'),custom_objects={'KerasLayer':hub.KerasLayer})




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetic')
def diabetic():
    return render_template('diabetic.html')

@app.route('/predictdiabetic',methods=['POST'])
def predict():
	
	if request.method == 'POST':
		preg = int(request.form['pregnancies'])
		glucose = int(request.form['glucose'])
		bp = int(request.form['bloodpressure'])
		st = int(request.form['skinthickness'])
		insulin = int(request.form['insulin'])
		bmi = float(request.form['bmi'])
		dpf = float(request.form['dpf'])
		age = int(request.form['age'])

		data = np.array([[preg,glucose,bp,st,insulin,bmi,dpf,age]])

		my_prediction = rfc_diabetic.predict(data)

		return render_template('diabetic.html',prediction = my_prediction)


## Brast Cancer
@app.route('/breastcancer')
def breastcancer():
    return render_template('breastcancer.html')

@app.route('/predictbc',methods=['POST'])
def predictbc():
	input_features = [float(x) for x in request.form.values()]
	features_value = [np.array(input_features)]

	features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']

	df = pd.DataFrame(features_value, columns=features_name)
	output = model_bc.predict(df)

	return render_template('breastcancer.html', prediction_text= output)


#Corona
@app.route('/corona')
def corona():
    return render_template('corona.html')

@app.route('/coronapredict',methods=['POST'])
def coronapredict():
	if request.method == "POST":
		myDict = request.form
		feaver = float(myDict['feaver'])
		age = int(myDict['age'])
		pain = int(myDict['pain'])
		runnynose = int(myDict['runnynose'])
		diffbreath = int(myDict['diffbreath'])

		feature=[feaver,pain,age,runnynose,diffbreath]
		
		feature_value = [np.array(feature)]
		feature_name=['feaver','bodypain','age','runnynose','diffbreath']
		df = pd.DataFrame(feature_value,columns=feature_name)

		infprob = clf_corona.predict(df)[0]

		return render_template('corona.html', inf=infprob)


#Heart
@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predictheart',methods=['POST'])
def heartpredict():

	if request.method == 'POST':
		age = int(request.form['age'])
		sex = int(request.form['sex'])
		cpt = int(request.form['cpt'])
		rbp = int(request.form['rbp'])
		sc = int(request.form['sc'])
		fbs = int(request.form['fbs'])
		rer = int(request.form['rer'])
		mra = int(request.form['mra'])
		eia = int(request.form['eia'])
		oldpeak = float(request.form['oldpeak'])
		slope = int(request.form['slope'])
		vessels = int(request.form['vessels'])
		thal = int(request.form['thal'])

		data = np.array([[age,sex,cpt,rbp,sc,fbs,rer,mra,eia,oldpeak,slope,vessels,thal]])

		scaler = StandardScaler()

		scaled = scaler.fit_transform(data)
		pred =  clf_heart.predict(scaled)

		return render_template('heart.html',heart_prediction = pred)



#Liver
@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predictliver',methods=['POST'])
def predictliver():
	
	if request.method == 'POST':
		age = int(request.form['age'])
		total_bilirubin = float(request.form['total_bilirubin'])
		direct_bilirubin = float(request.form['direct_bilirubin'])
		ap = int(request.form['alkaline_phosphotase'])
		aa = int(request.form['alamine_aminotransferase'])
		asa = int(request.form['aspartate_aminotransferase'])
		total_protiens = float(request.form['total_protiens'])
		albumin = float(request.form['albumin'])
		agratio = float(request.form['agratio'])

		data = np.array([[age,total_bilirubin,direct_bilirubin,
							ap,aa,asa,total_protiens,albumin,agratio]])


		scaler_data = StandardScaler()
		scaled_data = scaler_data.fit_transform(data)

		my_prediction = rfc_diabetic.predict(scaled_data)

		return render_template('liver.html',prediction = my_prediction)



def model_predict(image_path,model):
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = model.predict(image)

    if result[0]<=0.5:
        result = "Not Pneumonia"
        return result
    else:
        result = "Pneumonia"
        return result

IMG_FOLDER = os.path.join('static', 'upload')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

#Pneumonia
@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/predictpneumonia',methods=['GET','POST'])
def predictpneumonia():
	if request.method == 'POST':
		file = request.files['image'] # fetch input
		filename = file.filename 

		file_path = os.path.join('static/upload', filename)
		file.save(file_path)

		pred = model_predict(file_path,model_pneumonia)

		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)


		return render_template('pneumonia.html',predict = pred,user_image = full_filename)

if __name__ == '__main__':
	app.run(debug=True)