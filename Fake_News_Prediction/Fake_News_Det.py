from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from google import genai


app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

def generate_gemini_prediction(news):
    client = genai.Client(api_key="Put your own google geminai api key")
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents="A news string will be given below as a input to you, determine whether the news is Real or Fake by Analyzing all the available resources & news websites. return 'REAL' if the news is true return 'FAKE' if the news is false(only return true or false, no need to explain) : \n " + news
    )
    return response.text

@app.route('/')
def home():
     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print("Machine Learning Result : " + pred)
        gemini_result = 0;
        gemini_result = generate_gemini_prediction(message)
        print("Gemini API result:", gemini_result)
       
        model_pred_str = 'REAL' if pred[0] == 'REAL' else 'FAKE'
        gemini_result = gemini_result.strip()

        if model_pred_str == 'REAL' and gemini_result == 'REAL':
            final_pred = ['REAL']
        elif model_pred_str == 'REAL' and gemini_result == 'FAKE':
            final_pred = ['FAKE/REAL']
        elif model_pred_str == 'FAKE' and gemini_result == 'FAKE':
            final_pred = ['FAKE']
        elif model_pred_str == 'FAKE' and gemini_result == 'REAL':
            final_pred = ['REAL/FAKE']
        else:
            final_pred = ['ERROR']

        return render_template('index.html', prediction=final_pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
