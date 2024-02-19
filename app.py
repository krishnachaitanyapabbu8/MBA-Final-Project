from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictpipeline

application = Flask(__name__)

app = application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
    months_as_customer=int(request.form.get('months_as_customer')),
    age=int(request.form.get('age')),
    policy_number=int(request.form.get('policy_number')),
    policy_deductable=int(request.form.get('policy_deductable')),
    policy_annual_premium=float(request.form.get('policy_annual_premium')),
    umbrella_limit=int(request.form.get('umbrella_limit')),
    insured_zip=int(request.form.get('insured_zip')),
    capital_gains=int(request.form.get('capital_gains')),
    capital_loss=int(request.form.get('capital_loss')),
    incident_hour_of_the_day=int(request.form.get('incident_hour_of_the_day')),
    number_of_vehicles_involved=int(request.form.get('number_of_vehicles_involved')),
    bodily_injuries=int(request.form.get('bodily_injuries')),
    witnesses=int(request.form.get('witnesses')),
    total_claim_amount=float(request.form.get('total_claim_amount')),
    injury_claim=float(request.form.get('injury_claim')),
    property_claim=float(request.form.get('property_claim')),
    vehicle_claim=float(request.form.get('vehicle_claim')),
    auto_year=int(request.form.get('auto_year')),
    policy_bind_date=request.form.get('policy_bind_date'),
    policy_state=request.form.get('policy_state'),
    policy_csl=request.form.get('policy_csl'),
    insured_sex=request.form.get('insured_sex'),
    insured_education_level=request.form.get('insured_education_level'),
    insured_occupation=request.form.get('insured_occupation'),
    insured_hobbies=request.form.get('insured_hobbies'),
    insured_relationship=request.form.get('insured_relationship'),
    incident_date=request.form.get('incident_date'),
    incident_type=request.form.get('incident_type'),
    collision_type=request.form.get('collision_type'),
    incident_severity=request.form.get('incident_severity'),
    authorities_contacted=request.form.get('authorities_contacted'),
    incident_state=request.form.get('incident_state'),
    incident_city=request.form.get('incident_city'),
    incident_location=request.form.get('incident_location'),
    property_damage=request.form.get('property_damage'),
    police_report_available=request.form.get('police_report_available'),
    auto_make=request.form.get('auto_make'),
    auto_model=request.form.get('auto_model')
)

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        #print("Before Prediction")
        
        predict_pipeline=predictpipeline()
        #print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        #print("after Prediction")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0") 
