#Ah la la, Ah la la, Gimme Three wishes, I wanna be that Dirtyfinger and his six...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob
import nltk

import pyLDAvis
import pyLDAvis.sklearn

import numpy as np
import pandas as pd


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


def create_tfidf_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1)):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a tf-idf score for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram)
    doc_term_matrix = tfidf_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, tfidf_vect


def create_wordcount_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1)):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a wordcount for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, count_vect

#USE MATRIX AND VECTORS from last two funcitons as parameters for the next two functions.
# create_tfidf_matrix -> show_nmf_topic_words
# create_wordcount_matrix -> show_LDA_topic_words

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

def show_nmf_topic_words(matrix, vector, n_topics=5, n_words=5):
    """
    Accepts a doc_term_matrix and a fitted vectorizing model (from previous function). 
    Fits the NMF algorithm to the matrix.
    Uses the features/words from the vectorizing model.
    Prints the top n words for n topics.
    """
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(matrix )
    for i, topic in enumerate(nmf.components_): 
        print(f"Top {n_words} words for topic #{i}:") 
        print([vector.get_feature_names()[i] for i in topic.argsort()[-n_words:]]) 
        print('\n')


def find_top_documents_per_topic(lda_W, documents,n_docs):
    """
    lda_W is LDA.fit(matrix).transform(matrix)
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

def pyLDAvis(input_column, n_topics):
    matrix, vector = create_tfidf_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1))
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    pyLDAvis.sklearn.prepare(LDA, matrix, vector)

def show_pyLDAvis_dashboard(lda_fitted_model, doc_term_matrix, vector):
    pyLDAvis.sklearn.prepare(lda_fitted_model, doc_term_matrix, vector)


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