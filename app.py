from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv

app = Flask(__name__)

# Load the trained model from the updated pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Define a dictionary mapping numerical labels to medical condition names
medical_condition_map = {
    0: 'Arthritis',
    1: 'Asthma',
    2: 'Cancer',
    3: 'Diabetes',
    4: 'Hypertension',
    5: 'obesity'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    name = request.form.get('name')
    gender = request.form.get('gender')
    blood_group =request.form.get('blood-group')
    
    # Use the loaded model to predict the medical condition
    # Encode gender and blood group using LabelEncoder
    gender_encoded = label_encoder.fit_transform([gender])[0]
    blood_group_encoded = label_encoder.fit_transform([blood_group])[0]

    # Use the loaded model to predict the medical condition
    predicted_condition = model.predict(np.array([gender_encoded, blood_group_encoded]).reshape(1,-1))[0]
    result = medical_condition_map[predicted_condition]

    # Store the entry data in a CSV file
    with open('user_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, gender, blood_group, result])

    return render_template('result.html', name=name, predicted_condition=result)

if __name__ == '__main__':
    app.run(debug=True)
