import json
from enum import Enum

class DataLoader():
    
    def __init__(self, path):
        self.json_data = self.load_json(path)
    
    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def load_cate(self, cate):
        cate_data = [item for item in self.json_data if item["Category"] == cate]
        return cate_data
    
    
class Category(Enum):
    CyberCrime = "Cyber Crime"
    Discrimination = "Discrimination"
    FinacialCrime = "Financial Crime"
    Fraud = "Fraud"
    PoliticalSensitivity = "Political Sensitivity"
    PrivacyInvasion = "Privacy Invasion"
    PropertyViolation = "Property Violation"
    PublicThreat = "Public Threat"
    UnsuitableContent = "Unsuitable Content"
    ViolenceAndDamaging = "Violence and Damaging" 