from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob
import nltk

import pyLDAvis
import pyLDAvis.sklearn

import numpy as np
import pandas as pd

import unicodedata
import re


#Prepare

def create_biganswer_col(df):
    """
    Accepts a DataFrame as an input. The dataframe will be reduced to the columns that are of a string/object.
    Nulls will be made into a string value called 'n/a'. Each column is double-checked as a string type
    by using the .astype() method.
    The returned value is a Series that is all the column's values smashed into one column.
    """
    df_quals = df.select_dtypes('object')
    df_quals = df_quals.fillna('n/a')
    df_quals = df_quals.astype('str')
    col_names = list(df_quals.columns)
    list_of_cols = [df_quals[i] for i in col_names]
    big_answer = list_of_cols[13]
    for i in range(len(list_of_cols)-1):
        big_answer = big_answer + list_of_cols[i]
    return big_answer

#I would appened this big answer column back to the dataframe you are working with.
#Maybe just the df.select_dtypes('object'). That would be just the qualitative answers.

def lemmatize(text):
    """
    accept some text and return the text after applying lemmatization to each word.
    Use with .apply to a Pandas Series
    """
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    return ' '.join(lemmas)

def basic_clean(text):
    """
    Lowercase everything
    Normalize unicode characters
    Replace anything that is not a letter, number, whitespace or a single quote
    """
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    text = re.sub(r"[\r|\n|\r\n]+", ' ', text)
    return text


def set_stop_words(stop_words):
    """
    Takes a list of strings. These strings will be appended to the 'english' stop words used in sklearn
    objects that have a stopword parameter.
    Returns a very big list. Assign this to the stopwords parameter in the sklearn objects.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
    stopWords = stop_words + list(esw) 
    return stopWords

def create_wordcount_matrix(input_column, max_df=0.3, min_df=2, ngram=(1,3), stop_words='english'):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a wordcount for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, count_vect

def create_LDA_model(matrix, vector, n_topics=5):
    """
    Requires the matrix and vector created by the create_wordcount_matrix function.
    Input for the pyLDAvis exploration.
    """
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    return LDA


#Wordcounts or Ngrams Counts

# def find_word_counts(input_column, max_df=.3, min_df=2, ngram_range=(1,3), stop_words='english'):
#     """
#     Accepts a column of text. Uses default parameters for the WordCount Vectorizer.
#     Returns a dataframe with the word list, and their frequency as the other column.
#     """
#     input_column = input_column.dropna().apply(basic_clean)
#     input_column = input_column.apply(lemmatize)
#     cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)   
#     blob = cv.fit_transform(input_column)    
#     word_list = cv.get_feature_names()    
#     count_list = blob.toarray().sum(axis=0)
#     word_counts = {'word_list': word_list, 'count_list': count_list}
#     df_word_count = pd.DataFrame(data=word_counts)
#     return df_word_count

def find_word_counts(input_column, max_df=.3, min_df=2, ngram_range=(1,3), stop_words='english'):
#Cleans up nulls, weird characters, and reduces them to lemmas
    input_column = input_column.dropna().apply(basic_clean)
    input_column = input_column.apply(lemmatize)

#cv is the Vector, cv_fit is the Matrix
    cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)   
    cv_fit=cv.fit_transform(input_column)
    
#An array of all the words that make up the Vector
    word_list = cv.get_feature_names()    
#An array of the freq-counts of each of the words in word_list
#   .toarray() is a matrix that is as wide as the word list, tall as the amount of rows.  
#   sum(axis=0) collapses the matrix into a vector, summing each column into one value. It will be as wide as word_list
    count_list = cv_fit.toarray().sum(axis=0)

#The dictionary that will define what the DataFrame will be shaped
    word_counts = {'word_list': word_list, 'count_list': count_list}
    df_word_count = pd.DataFrame(data=word_counts)
    return df_word_count

#Topic Modeling

def show_LDA_topic_words(matrix, vector, n_topics=5, n_words=5):
    """
    Accepts a doc_term_matrix and a fitted vectorizing model (from previous function). 
    Fits the LDA algorithm to the matrix.
    Uses the features/words from the vectorizing model.
    Prints the top n words for n topics.
    """
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    for i, topic in enumerate(LDA.components_): 
        print(f"Top {n_words} words for topic #{i}:") 
        print([vector.get_feature_names()[i] for i in topic.argsort()[-n_words:]]) 
        print('\n')

def assign_topic_column(input_column, max_df=.8, min_df=2, stop_words='english', ngram_range=(1,3), n_components=3):
    """
    takes a pandas column/Series. Runs through the topic modeling pipeline. Assigns the topic_id
    to each row. topic_id with highest probability is assigned.
    Returns a Series that is a int value of the topic (0--n)
    """
    input_column = input_column.astype('str')
    input_column = input_column.fillna('nan')
    input_column = input_column.apply(basic_clean)
    input_column = input_column.apply(lemmatize)
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
    LDA.fit(doc_term_matrix)
    lda_H = LDA.transform(doc_term_matrix)
    topic_doc_df = pd.DataFrame(lda_H)
    return topic_doc_df.idxmax(axis=1), topic_doc_df.max(axis=1)

def assign_topic_column_probability(input_column, max_df=.8, min_df=2, stop_words='english', ngram_range=(1,3), n_components=3):
    """
    takes a pandas column/Series. Runs through the topic modeling pipeline.
    Returns a Series that is a probability score for the most likely topic.
    """
    input_column = input_column.fillna('nan')
    input_column = input_column.astype('str')
    input_column = input_column.apply(basic_clean)
    input_column = input_column.apply(lemmatize)
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
    LDA.fit(doc_term_matrix)
    lda_H = LDA.transform(doc_term_matrix)
    topic_doc_df = pd.DataFrame(lda_H)
    return topic_doc_df.max(axis=1)

def find_top_documents_per_topic(lda_W, documents,n_docs):
    """
    lda_W is LDA.fit(matrix).transform(matrix)
        or the output of create_LDA_model().transform(matrix from create_wordcount_matrix)
    n_top_docs is a integer for how many documents you want to represent a topic
    """
    df_W = pd.DataFrame(lda_W, index=documents.index)
    for column in df_W:
        indexes = df_W[column].sort_values().tail(n_docs).index
        print(f'Top {n_docs} Documents for Topic {column}: \n')
        counter = 1
        for i in indexes:
            print(f"Document {counter}")
            print(documents[i] + '\n')
            counter += 1

#TOPIC EXPLORATION with pyLDAvis

def show_pyLDAvis_dashboard2(input_column, n_topics):
    """
    Requires just the question column to create the dashboard.
    """
    matrix, vector = create_wordcount_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1))
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    pyLDAvis.sklearn.prepare(LDA, matrix, vector)

def show_pyLDAvis_dashboard(lda_fitted_model, doc_term_matrix, vector):
    """
    Requires the create_LDA_model and create_wordcount_matrix outputs as parameters.
    """
    pyLDAvis.sklearn.prepare(lda_fitted_model, doc_term_matrix, vector)

#Sentiment Analysis

def find_polarity(input_text):
    """
    Takes one document and outputs a polarity score. -1 is Very Negative, 1 is Super Positive
    This function should be applied to a column and outputted to a new column.
    """
    return TextBlob(input_text).sentiment.polarity


def find_subjectivity(input_text):
    """
    Takes one document and outputs a subjectivity score. 0 is cold objectivity, 1 is very subjective
    This function should be applied to a column and outputted to a new column.
    """
    return TextBlob(input_text).sentiment.subjectivity


def create_sentiment_df(input_column):
    """
    Accepts a column of text data. Returns a dataframe with the original text with their polarity and 
    subjectivity scores appended to it.
    """
    cols = ['text','polarity','subjectivity']
    df = pd.DataFrame(columns=cols)
    df['text'] = input_column
    df['polarity'] = input_column.apply(find_polarity)
    df['subjectivity'] = input_column.apply(find_subjectivity)
    return df