import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter

from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #Plot 1 - Data for plotting Distribution of Message Categories
    #Get names of all the categories
    category_names = df.columns[4:]
    
    #Unpivot the DataFrame from wide format to long format. Create 2 columns for category names and values instead of the 36 category columns
    df_plot = df.melt(id_vars=['id','message','original','genre'],var_name='Category')
    
    #Keep the rows with Value 1
    df_plot = df_plot[df_plot['value'] == 1]
    
    #Group the data by categories and count of messages per category
    category_count=df_plot.groupby(['Category']).count()['message'].sort_values(ascending=False)
    
    #Plot 2 - Data for plotting Precision-Recall of the model
    #Create X - feature set
    X = df['message'].values
    
    #Create Y - target set containing all the category columns
    Y = df.drop(['id','message','original','genre'], axis=1)
    
    #Split the df into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    #Predict for X_test
    y_pred = model.predict(X_test)
    y_scores = {'precisions':[],
               'recalls':[],
               'f_scores':[]}
    
    #Get the Scores for each category 
    for i in range(36):
        scores = precision_recall_fscore_support(y_test.values[:,i], y_pred[:,i],average='weighted')
        y_scores['precisions'].append(scores[0])
        y_scores['recalls'].append(scores[1])
        y_scores['f_scores'].append(scores[2])
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x= y_scores['recalls'],
                    y= y_scores['precisions'],
                    mode = 'markers'
                )
            ],

            'layout': {
                'title': 'Recall - Precision Plot',
                'yaxis': {
                    'title': "Precision",
                    'tick0': 0,
                    'dtick': 0.2,
                    'tickcolor': '#000',
                    'range': [0,1]
                },
                'xaxis': {
                    'title': "Recall",
                    'tick0': 0,
                    'dtick': 0.2,
                    'tickcolor': '#000',
                    'range': [0,1]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
