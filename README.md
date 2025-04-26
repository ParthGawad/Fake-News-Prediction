# Fake-News-Prediction
A Project about Fake News Prediction Using AI &amp; ML

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 

## Intro : 
The Rise of Fake News The spread of fake news has become
prevalent in recent years, with the growth of social media,
online news outlets, and the viral nature of information. Nearly
half of india’s internet users, obtain their news from unreliable
sources, & turns a blind to truthness of the news .

Linguistic patterns, sentiment analysis, and credibility assessments using Machine Learning (ML) algorithms, especially Natural Language Processing (NLP) basedmodels have
also produced promising results in the detection of fake news.

## Requirements : 
- Python 3.X
- flask (python import)
- sklearn’s Tfidvectorizer (python import)
- sklearn’s PassiveAggressiveClassifier (python import)
- Pickle (python import)
- pandas (python import)
- google genai (python import)

## Usage : 
- Clone my repository.
- Open CMD in working directory.
- Run `pip install -r requirements.txt`
- Open project in any IDE(Pycharm or VSCode)
- Run `Fake_News_Det.py`, go to the `http://127.0.0.1:5000/`
- If you want to build your model with the some changes, you can check the `Fake_News_Detection.ipynb`.
- DISCLAIMER Sometimes predictions may be wrong because of history of dataset being used to train the model. Because of this drawback, i have implemented google geminai api as a backup if the model prediction returns false positive.
- Before running the project, remember to unzip the 'news.csv' dataset file, or else the code will return errors.

## How It Works
### Model Training Process

- Dataset: Uses labeled news data from news.csv containing both fake and real news articles
- Text Processing: Implements TF-IDF vectorization to convert text to numerical features
- Removes English stop words
-  Ignores terms appearing in more than 70% of documents


### Model: 
- Uses PassiveAggressiveClassifier, which is:
An online learning algorithm suitable for large-scale learning
"Passive" when prediction is correct (keeping the model unchanged)
"Aggressive" when prediction is wrong (updating to fix misclassification)

- Performance: Achieves 93.69% accuracy on test data

### Web Application
- Framework: Built with Flask
- User Interface: Simple form for users to input news text
- Prediction Process: User submits a news article,System processes text through the trained model
In parallel, sends the text to Google's Gemini API
Compares both results to determine final classification
Displays results to the user



### Combined Decision Logic
The system uses both predictions to provide a nuanced result:

- Both say REAL → Final prediction is REAL
- Both say FAKE → Final prediction is FAKE
- Conflicting results → Indicates uncertainty (FAKE/REAL or REAL/FAKE)

## WorkFlow Diagram :
![image](https://github.com/user-attachments/assets/9c4290a2-f9f6-4b78-8f37-940635dd9714)

## Output ScreenShots : 
![Screenshot 2025-04-20 005723](https://github.com/user-attachments/assets/02d3914a-7e9d-46e5-a4bc-935be812a68e)
![Screenshot 2025-04-20 010021](https://github.com/user-attachments/assets/954d602b-2311-446a-b24b-23216ef2ea56)
![Screenshot 2025-04-20 010121](https://github.com/user-attachments/assets/7ca3c706-6d99-4ad0-ba54-a4258c762c49)


