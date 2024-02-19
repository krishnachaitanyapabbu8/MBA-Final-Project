import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictpipeline:
    def __init__(self):
        pass

def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            #print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            #print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 months_as_customer: int,
                 age: int,
                 policy_number: int,
                 policy_deductable: int,
                 policy_annual_premium: float,
                 umbrella_limit: int,
                 insured_zip: int,
                 capital_gains: int,
                 capital_loss: int,
                 incident_hour_of_the_day: int,
                 number_of_vehicles_involved: int,
                 bodily_injuries: int,
                 witnesses: int,
                 total_claim_amount: float,
                 injury_claim: float,
                 property_claim: float,
                 vehicle_claim: float,
                 auto_year: int,
                 policy_bind_date: str,
                 policy_state: str,
                 policy_csl: str,
                 insured_sex: str,
                 insured_education_level: str,
                 insured_occupation: str,
                 insured_hobbies: str,
                 insured_relationship: str,
                 incident_date: str,
                 incident_type: str,
                 collision_type: str,
                 incident_severity: str,
                 authorities_contacted: str,
                 incident_state: str,
                 incident_city: str,
                 incident_location: str,
                 property_damage: str,
                 police_report_available: str,
                 auto_make: str,
                 auto_model: str,
                 fraud_reported: str):

        self.months_as_customer = months_as_customer
        self.age = age
        self.policy_number = policy_number
        self.policy_deductable = policy_deductable
        self.policy_annual_premium = policy_annual_premium
        self.umbrella_limit = umbrella_limit
        self.insured_zip = insured_zip
        self.capital_gains = capital_gains
        self.capital_loss = capital_loss
        self.incident_hour_of_the_day = incident_hour_of_the_day
        self.number_of_vehicles_involved = number_of_vehicles_involved
        self.bodily_injuries = bodily_injuries
        self.witnesses = witnesses
        self.total_claim_amount = total_claim_amount
        self.injury_claim = injury_claim
        self.property_claim = property_claim
        self.vehicle_claim = vehicle_claim
        self.auto_year = auto_year
        self.policy_bind_date = policy_bind_date
        self.policy_state = policy_state
        self.policy_csl = policy_csl
        self.insured_sex = insured_sex
        self.insured_education_level = insured_education_level
        self.insured_occupation = insured_occupation
        self.insured_hobbies = insured_hobbies
        self.insured_relationship = insured_relationship
        self.incident_date = incident_date
        self.incident_type = incident_type
        self.collision_type = collision_type
        self.incident_severity = incident_severity
        self.authorities_contacted = authorities_contacted
        self.incident_state = incident_state
        self.incident_city = incident_city
        self.incident_location = incident_location
        self.property_damage = property_damage
        self.police_report_available = police_report_available
        self.auto_make = auto_make
        self.auto_model = auto_model
        self.fraud_reported = fraud_reported

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "months_as_customer": [self.months_as_customer],
                "age": [self.age],
                "policy_number": [self.policy_number],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
                "umbrella_limit": [self.umbrella_limit],
                "insured_zip": [self.insured_zip],
                "capital_gains": [self.capital_gains],
                "capital_loss": [self.capital_loss],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses],
                "total_claim_amount": [self.total_claim_amount],
                "injury_claim": [self.injury_claim],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],
                "auto_year": [self.auto_year],
                "policy_bind_date": [self.policy_bind_date],
                "policy_state": [self.policy_state],
                "policy_csl": [self.policy_csl],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_occupation": [self.insured_occupation],
                "insured_hobbies": [self.insured_hobbies],
                "insured_relationship": [self.insured_relationship],
                "incident_date": [self.incident_date],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "incident_state": [self.incident_state],
                "incident_city": [self.incident_city],
                "incident_location": [self.incident_location],
                "property_damage": [self.property_damage],
                "police_report_available": [self.police_report_available],
                "auto_make": [self.auto_make],
                "auto_model": [self.auto_model],
                "fraud_reported": [self.fraud_reported],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
