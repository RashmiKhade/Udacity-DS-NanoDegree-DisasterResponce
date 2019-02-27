# import libraries
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Load the data from database. 
    Define feature and target variables X and Y
    X are the messages and Y are all the category columns
    INPUT:
    database_filepath - path of the database file
    
    OUTPUT:
    X - feature variable
    Y - target variable
    category_names - names of categories
    """
    
    # load data from database
    url ='sqlite:///'+database_filepath
    engine = create_engine(url)
    df = pd.read_sql_table('DisasterMessages',con=engine)
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names
    

def tokenize(text):
    """ 
    Split the message and clean the tokens
        1. Tokenize text using word_tokenize
        2. Lemmetize the tokens
        3. Normalize the tokens
        4. Strip any white spaces
    
    INPUT - Disaster Message
    OUTPUT - list of processed tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """ 
    Build a machine learning pipeline
        
    INPUT - None
    OUTPUT - Pipeline of optimized model
    """
    # Pipeline of CountVextorizer, TfdifTransformer and MultiOutputClassifier
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators': [50, 30],
              'clf__estimator__min_samples_split': [3, 2] 
    }
    
    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=2, n_jobs=4)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """ 
    Evaluate the model against the test dataset.
    Report the f1 score, precision and recall for each output category of the dataset. 
        
    INPUT - 
    model - Pipeline of optimized model
    X_test - test feature dataset
    y_test - test target dataset
    category_names - names of vategories
    
    OUTPUT - None
    """
    # Predict for test set
    y_pred = model.predict(X_test)
    
    print("**** Scores for each category *****\n")
    for i in range(36):
        print("Scores for '{}':".format(category_names[i]))
        print(classification_report(y_test.values[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """ 
    Save the model as a pickle file.
        
    INPUT - 
    model - Pipeline of optimized model
    model_filepath - path to save the file
    
    OUTPUT - None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
