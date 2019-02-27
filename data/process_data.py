import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from csv into dataframes and merge the 2 dataframes.
    Return this merged dataframe.
    INPUT:
    messages_filepath - path of the file containing the messages
    categories_filepath - path of the file containing the categories
    
    OUTPUT:
    df - merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')

    return df

def clean_data(df):
    """
    Split the categories column into multiple columns, 1 for each category (separated by ;)
    Remove duplicates and return this cleaned dataframe.
    INPUT:
    df - merged dataframe of messages and categories
    
    OUTPUT:
    df - merged dataframe
    """
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str[:-2]
  
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # Replace `categories` column in `df` with new category columns.
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1,errors='ignore')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # Remove duplicates.
    df = df.drop_duplicates()   
    
    return df

def save_data(df, database_filename):
    
    """
    Save the data to database.
    
    INPUT:
    df - cleaned dataframe to be stored in database
    database_filename - path of the database file
    
    """
    url ='sqlite:///'+database_filename
    engine = create_engine(url)
    engine.execute("DROP TABLE IF EXISTS DisasterMessages")

    df.to_sql('DisasterMessages', engine, index=False)
    #print(engine.execute("SELECT * FROM DisasterMessages").fetchone() ) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
