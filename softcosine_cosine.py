from sklearn.datasets import fetch_20newsgroups
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import re
import math

import gensim
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from gensim import corpora

import gensim.downloader as api
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity

# create stopword list
# stoplist = data_prep.build_stopword_list()
stoplist = stopwords.words("english")
# using glove-wiki dataset to  create word2vec model
# w2v_model = api.load("glove-wiki-gigaword-100")

# Loads the google pre-trained word2vec model once into RAM and this class
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 

# Loads the ConceptNet pre-trained word2vec model once into RAM and this class
# https://github.com/commonsense/conceptnet-numberbatch
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('numberbatch-en.txt', binary=False)

print('Word2vec model created!')
# w2v_model.save("word2vec.model")

def get_data():
    category = ['alt.atheism', 'rec.autos', 'comp.sys.mac.hardware', 'rec.sport.hockey', 'sci.space']
    # get data via sklearn
    train = fetch_20newsgroups(subset="train", categories=category)
    test = fetch_20newsgroups(subset="test", categories=category)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # define single step preprocess function
    look_up = np.vectorize(lambda x: train.target_names[x])
    remove_header = lambda x: re.split(r'[\n]{2}', x, maxsplit=1)[1]
    remove_chars = lambda x: re.sub('[^a-zA-Z0-9 _]+', '', x)
    simple_preprocess = lambda x: " ".join([lemmatizer.lemmatize(word)
                                            for word in x.split()
                                            if word not in stoplist and
                                            2 < len(word) < 20])
    # bring all the preprocess function together
    complete = np.vectorize(lambda x: simple_preprocess(remove_chars(remove_header(x))))

    # create the DataFrame
    data_frame = pd.DataFrame({"id": list(range(len(test.target) + len(train.target))),
                    "text": np.hstack((complete(train.data), complete(test.data))), 
                    "newsgroup": np.hstack((look_up(train.target), look_up(test.target))), 
                    # "file_location": np.hstack((train.filenames, test.filenames))
                    })
    print('Data preprocessing completed!')
    return data_frame

# Function to calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

# Function to calculate soft cosine similarity
def calculate_softcosine_w2v(test_data):
    data = [i.split() for i in (test_data.text).tolist()]
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(d) for d in data]
    
    similarity_index = WordEmbeddingSimilarityIndex(w2v_model)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
    
    softsim_w2v_matrix =  np.empty(shape=(len(data), len(data))) * np.nan
    for d1 in range(0, len(data)):
        for d2 in range(0, len(data)):
            softsim_w2v_matrix[d1, d2] = similarity_matrix.inner_product(corpus[d1], corpus[d2], normalized=True)
   
    doc_sim_max_index, doc_sim_max_values = calculate_max_similarity(softsim_w2v_matrix)
    softsim_w2v_df = export_result(test_data, doc_sim_max_index, doc_sim_max_values, 'softsim_w2v')
    print("Similarity using soft cosine similarity using w2v vectors is calculated!!")
    return softsim_w2v_df

def calculate_cosine_tfidf(test_data):
    tfidf = TfidfVectorizer()
    tfidf_vectors = tfidf.fit_transform(test_data.text)

    cosine_tfidf_matrix = np.zeros(shape=(len(test_data), len(test_data)))

    for index, array in enumerate(tfidf_vectors.toarray()):
        for index1, array1 in enumerate(tfidf_vectors.toarray()):
            cosine_tfidf_matrix[index, index1] = cosine_similarity(array, array1)

    doc_sim_max_index, doc_sim_max_values = calculate_max_similarity(cosine_tfidf_matrix)
    cosine_tfidf_df = export_result(test_data, doc_sim_max_index, doc_sim_max_values, 'cosine_tfidf')
    print("Similarity using cosine similarity using tfidf vectors is calculated!!")
    return cosine_tfidf_df
        

# Function to calculate word2vec vectors for each word of document
def vectorize_w2v(document):
    word_vecs = []
    for word in document:
        try:
            word_vecs.append(w2v_model[word])
        except KeyError:
            word_vecs.append(np.zeros(w2v_model.vector_size))
            # pass
    return np.mean(word_vecs, axis=0) if len(word_vecs) > 0 else np.nan

# Function to calculate cosine similarity
def calculate_cosine_w2v(test_data):
    data = [i.split() for i in (test_data.text).tolist()]
    texts_w2v = []
    # vectorize the documents from n-gram vectors into word2vec vectors
    if len(texts_w2v) < 1:
        texts_w2v = [vectorize_w2v(d) for d in data]
    
    cosine_w2v_matrix = np.zeros(shape=(len(data), len(data)))
    
    for d1 in range(0, len(data)):
        for d2 in range(0, len(data)):
            cosine_w2v_matrix[d1, d2] = sklearn.metrics.pairwise.cosine_similarity([texts_w2v[d1]], [texts_w2v[d2]])
    
    doc_sim_max_index, doc_sim_max_values = calculate_max_similarity(cosine_w2v_matrix)
    cosine_w2v_df = export_result(test_data, doc_sim_max_index, doc_sim_max_values, 'cosine_w2v')
    print("Similarity using cosine similarity using w2v is calculated!!")
    return cosine_w2v_df

# Function to output the index and value giving maximum similarity 
def calculate_max_similarity(similarity_matrix):
    similarity_matrix[similarity_matrix >= 0.999] = 0
    similarity_matrix_max_index = np.argmax(similarity_matrix, axis=1)
    similarity_matrix_max_values=np.around(similarity_matrix.max(axis=1), decimals=3)
    return similarity_matrix_max_index, similarity_matrix_max_values

# Function to append the result with intial dataframe
def export_result(documents, index, value, metric):
    # adding similarity result in same dataframe
    result_df = pd.DataFrame(value, columns=['document_similarities_max_values_{m}'.format(m=metric)])
    
    similar_document = []
    category = []
    for i in index:
        similar_document.append(documents.text.iloc[i])
        category.append(documents.newsgroup.iloc[i])
    
    result_df['similar_document_{m}'.format(m=metric)] = [i for i in similar_document]
    result_df['similar_newsgroup_{m}'.format(m=metric)] = [i for i in category]

    # exporting result into csv
    # result_df.to_csv('export/test_similarity_{type}.csv'.format(type='jss' if mode =='jss' else 'wmd'), sep=';', index=False)
    return result_df

# Calculating the accuracy of each method
def check_newsgroup(df):
    correct_count_softsim_w2v = 0 
    correct_count_cosine_w2v = 0
    correct_count_cosine_tfidf = 0

    for i in range(df.shape[0]):
        if df['newsgroup'][i] == df['similar_newsgroup_softsim_w2v'][i]:
            correct_count_softsim_w2v = correct_count_softsim_w2v + 1
        if df['newsgroup'][i] == df['similar_newsgroup_cosine_w2v'][i]:
            correct_count_cosine_w2v = correct_count_cosine_w2v + 1
        if df['newsgroup'][i] == df['similar_newsgroup_cosine_tfidf'][i]:
            correct_count_cosine_tfidf = correct_count_cosine_tfidf + 1

    accuracy_softsim_w2v = (correct_count_softsim_w2v/df.shape[0]) * 100
    accuracy_cosine_w2v = (correct_count_cosine_w2v/df.shape[0]) * 100
    accuracy_cosine_tfidf = (correct_count_cosine_tfidf/df.shape[0]) * 100

    return accuracy_softsim_w2v, accuracy_cosine_w2v, accuracy_cosine_tfidf

# Main function
def main():
    data = get_data()
    df = data[:200]
    print('Now, calculating similarities!')
    df1 = calculate_softcosine_w2v(df)
    df2 = calculate_cosine_w2v(df)
    df3 = calculate_cosine_tfidf(df)
    result_df = pd.concat([df, df1, df2, df3], axis=1)
    result_df.to_csv('test_similarity_output_0.csv', sep=';', index=False)
    acc_softsim_w2v, acc_cosine_w2v, acc_coisne_tfidf = check_newsgroup(result_df)
    print("The accuracy from {0} is: {1} , {2} is: {3}, {4} is: {5}".format('softsim_w2v', acc_softsim_w2v, 'cosine_w2v', acc_cosine_w2v, 'cosine_tfidf', acc_coisne_tfidf))
    
main()
