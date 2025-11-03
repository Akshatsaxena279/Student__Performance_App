from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

def get_advice(features, result):
    advice = []
    if result == 'FAIL':
        if features['study_hours'] < 2:
            advice.append('Your study hours are very low. You need to dedicate more time to studying to pass.')
        if features['attendance'] < 75:
            advice.append('Your attendance is low. Attending classes is crucial for understanding the material.')
        if features['internal_marks'] < 15:
            advice.append('Your internal marks are low. Focus on improving your performance in internal assessments.')
        if features['assignment_score'] < 5:
            advice.append('Your assignment scores are low. Assignments are a key part of the learning process.')
        if not advice:
            advice.append('There are multiple areas where you can improve. Focus on increasing your study hours and attendance.')
    else:
        advice.append('You are on the right track! Keep up the good work and maintain your current study habits.')
        
    return advice

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        student_name = request.form['student_name']
        attendance = float(request.form['attendance'])
        study_hours = float(request.form['study_hours'])
        internal_marks = float(request.form['internal_marks'])
        assignment_score = float(request.form['assignment_score'])

        features_dict = {
            'attendance': attendance,
            'study_hours': study_hours,
            'internal_marks': internal_marks,
            'assignment_score': assignment_score
        }

        # Rule-based checks for immediate failure
        if study_hours < 1 or attendance < 50:
            result = 'FAIL'
        else:
            features = np.array(list(features_dict.values())).reshape(1, -1)
            prediction = model.predict(features)
            result = 'PASS' if prediction[0] == 1 else 'FAIL'

        advice = get_advice(features_dict, result)

        prediction_data = {'student_name': [student_name], **features_dict, 'prediction': [result]}
        df = pd.DataFrame(prediction_data)

        if not os.path.isfile('predictions.csv'):
            df.to_csv('predictions.csv', index=False)
        else:
            df.to_csv('predictions.csv', mode='a', header=False, index=False)

        return render_template('index.html', 
                               prediction_text=f'Prediction for {student_name}: {result}',
                               student_name=student_name,
                               attendance=attendance, 
                               study_hours=study_hours, 
                               internal_marks=internal_marks, 
                               assignment_score=assignment_score,
                               advice=advice)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

@app.route('/history')
def history():
    if os.path.isfile('predictions.csv'):
        df = pd.read_csv('predictions.csv')
        return render_template('history.html', students=df.to_dict(orient='records'))
    return render_template('history.html', students=[])

@app.route('/edit/<student_name>', methods=['GET', 'POST'])
def edit_student(student_name):
    if not os.path.isfile('predictions.csv'):
        return redirect(url_for('history'))

    df = pd.read_csv('predictions.csv')
    student_data_df = df[df['student_name'] == student_name]

    if student_data_df.empty:
        return redirect(url_for('history'))

    if request.method == 'POST':
        df = pd.read_csv('predictions.csv')
        idx = df[df['student_name'] == student_name].index[0]

        df.loc[idx, 'student_name'] = request.form['student_name']
        df.loc[idx, 'attendance'] = float(request.form['attendance'])
        df.loc[idx, 'study_hours'] = float(request.form['study_hours'])
        df.loc[idx, 'internal_marks'] = float(request.form['internal_marks'])
        df.loc[idx, 'assignment_score'] = float(request.form['assignment_score'])
        df.loc[idx, 'prediction'] = request.form['prediction']

        df.to_csv('predictions.csv', index=False)
        return redirect(url_for('history'))

    student_data = student_data_df.iloc[0].to_dict()
    return render_template('edit.html', student=student_data)

@app.route('/delete/<student_name>')
def delete_student(student_name):
    if os.path.isfile('predictions.csv'):
        df = pd.read_csv('predictions.csv')
        df = df[df['student_name'] != student_name]
        df.to_csv('predictions.csv', index=False)
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
