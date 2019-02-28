# Disaster Response Pipeline Project

# Background:
Following a disaster, there are a number of different problems that may arise. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, so predictive modeling can help classify different messages more efficiently.

# About the dataset - Multilingual Disaster Response Messages
Figure Eight (https://www.figure-eight.com/) has provided a Disaster Response dataset which contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety leaving 26028 messages.

# Building a model
In this project, I built an ETL pipeline that cleaned messages using regex and NLTK. The text data was trained on a multioutput classifier model using random forest. The final deliverable is Flask app that classifies input messages and shows visualizations of key statistics of the dataset.

The random forest classifier model scored 93% on precision, 94% on recall and 92% on f1-score , after tuning the parameters using GridSearchCV.

# Files:
disaster_messages.csv - CSV containing the disasater messages

disaster_categories.csv - CSV containing category information for all the disaster messages

process_data.py - ETL Pipeline for data prep and storage

train_classifier.py - Text processing and machine learning pipeline that trains and tunes a model pipeline

run.py - Flask Web App for data visualizations and predicting categories for a Disasater Messages

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

