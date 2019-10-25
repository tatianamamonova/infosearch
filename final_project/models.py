import re
import pymorphy2
import pickle
import os
import numpy as np
import pandas as pd
from math import log
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine


def create_db_tfidf(file_name, path_to_db):
    documents1 = []
    documents2 = []
    related = {}
    
    morph = pymorphy2.MorphAnalyzer()
    df = pd.read_csv("quora.csv")
    
    for i, row in df[:5000].iterrows():
        id1 = "q"+str(i)
        id2 = "d"+str(i)

        doc1 = str(row["question1"]).lower()
        doc1 = re.split(r"[^а-яё]+", doc1)
        doc1 = [morph.parse(word)[0].normal_form for word in doc1]
        documents1.append(doc1)

        doc2 = str(row["question2"]).lower()
        doc2 = re.split(r"[^а-яё]+", doc2)
        doc2 = [morph.parse(word)[0].normal_form for word in doc2]
        documents2.append(doc2)

        if row["is_duplicate"] == 1:
            if id1 not in related:
                related[id1] = []
            related[id1].append(id2)
    
    corpus = [" ".join(document) for document in documents2]
    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names()
    
    path = os.path.join(path_to_db, "tfidf")
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    with open(os.path.join(path, "documents.pickle"), "wb") as pickle_file:
        pickle.dump(documents2, pickle_file)
    
    with open(os.path.join(path, "vocabulary.pickle"), "wb") as pickle_file:
        pickle.dump(vocabulary, pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "wb") as pickle_file:
        pickle.dump(X, pickle_file)


def search_tfidf(req, path_to_db):
    path = os.path.join(path_to_db, "tfidf")
    
    with open(os.path.join(path, "documents.pickle"), "rb") as pickle_file:
        documents = pickle.load(pickle_file)
    
    with open(os.path.join(path, "vocabulary.pickle"), "rb") as pickle_file:
        vocabulary = pickle.load(pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "rb") as pickle_file:
        data = pickle.load(pickle_file)
    
    morph = pymorphy2.MorphAnalyzer()
    
    req = req.lower()
    req = re.split(r"[^а-яё]+", req)
    req = [morph.parse(word)[0].normal_form for word in req]
    
    mul = [[0] for i in range(len(vocabulary))]
    
    for i in range(len(req)):
        if req[i] in vocabulary:
            mul[vocabulary.index(req[i])] = [1]
    
    result = [res[0] for res in np.dot(csr_matrix(data), csr_matrix(mul)).toarray()]
    result_dict = {" ".join(documents[i]): result[i] for i in range(len(result)) if result[i] != 0}
    output = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    
    docs = [o[0] for o in output[:10]]
    scores = [o[1] for o in output[:10]]
    
    return docs, scores


def tf(word, document):
    c = Counter(document)
    return c[word]/len(document)


def create_db_bm25(file_name, path_to_db):
    b = 0.75
    k = 2.0
    documents1 = []
    documents2 = []
    related = {}
    
    morph = pymorphy2.MorphAnalyzer()
    df = pd.read_csv("quora.csv")
    
    for i, row in df[:5000].iterrows():
        id1 = "q"+str(i)
        id2 = "d"+str(i)

        doc1 = str(row["question1"]).lower()
        doc1 = re.split(r"[^а-яё]+", doc1)
        doc1 = [morph.parse(word)[0].normal_form for word in doc1]
        documents1.append(doc1)

        doc2 = str(row["question2"]).lower()
        doc2 = re.split(r"[^а-яё]+", doc2)
        doc2 = [morph.parse(word)[0].normal_form for word in doc2]
        documents2.append(doc2)

        if row["is_duplicate"] == 1:
            if id1 not in related:
                related[id1] = []
            related[id1].append(id2)
    
    avgdl = np.mean([len(document) for document in documents2])
    
    idf = {}
    for document in documents2:
        for word in set(document):
            if word not in idf:
                idf[word] = 1
            else:
                idf[word] += 1
    idf = {word: log((len(documents2)-idf[word]+0.5)/(idf[word]+0.5)) for word in idf}
    
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for document in documents2:
        for word in document:
            index = vocabulary.setdefault(word, len(vocabulary))
            indices.append(index)
            data.append(idf[word]*tf(word, document)*(k+1)/(tf(word, document)+k*(1-b+b*len(document)/avgdl)))
        indptr.append(len(indices))
    
    path = os.path.join(path_to_db, "bm25")
    
    data = csr_matrix((data, indices, indptr))
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    with open(os.path.join(path, "documents.pickle"), "wb") as pickle_file:
        pickle.dump(documents2, pickle_file)
    
    with open(os.path.join(path, "vocabulary.pickle"), "wb") as pickle_file:
        pickle.dump(vocabulary, pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def search_bm25(req, path_to_db):
    path = os.path.join(path_to_db, "bm25")
    
    with open(os.path.join(path, "documents.pickle"), "rb") as pickle_file:
        documents = pickle.load(pickle_file)
    
    with open(os.path.join(path, "vocabulary.pickle"), "rb") as pickle_file:
        vocabulary = pickle.load(pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "rb") as pickle_file:
        data = pickle.load(pickle_file)
    
    morph = pymorphy2.MorphAnalyzer()
    
    req = req.lower()
    req = re.split(r"[^а-яё]+", req)
    req = [morph.parse(word)[0].normal_form for word in req]
    
    mul = [[0] for i in range(len(vocabulary))]
    
    for i in range(len(req)):
        if req[i] in vocabulary:
            mul[vocabulary[req[i]]] = [1]
    
    result = [res[0] for res in np.dot(data, csr_matrix(mul)).toarray()]
    result_dict = {" ".join(documents[i]): result[i] for i in range(len(result)) if result[i] != 0}
    output = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    
    docs = [o[0] for o in output[:10]]
    scores = [o[1] for o in output[:10]]
    
    return docs, scores


def create_db_fasttext(file_name, path_to_db):
    documents1 = []
    documents2 = []
    related = {}
    
    morph = pymorphy2.MorphAnalyzer()
    df = pd.read_csv("quora.csv")
    
    for i, row in df[:5000].iterrows():
        id1 = "q"+str(i)
        id2 = "d"+str(i)

        doc1 = str(row["question1"]).lower()
        doc1 = re.split(r"[^а-яё]+", doc1)
        doc1 = [morph.parse(word)[0].normal_form for word in doc1]
        documents1.append(doc1)

        doc2 = str(row["question2"]).lower()
        doc2 = re.split(r"[^а-яё]+", doc2)
        doc2 = [morph.parse(word)[0].normal_form for word in doc2]
        documents2.append(doc2)

        if row["is_duplicate"] == 1:
            if id1 not in related:
                related[id1] = []
            related[id1].append(id2)
    
    model_file = './fasttext/model.model'
    model = KeyedVectors.load(model_file)
    
    dimensions = model.vector_size
    
    data = []
    
    for document in documents2:
        doc_embedding = np.array([0 for i in range(dimensions)], dtype="float64")
        for word in document:
            if word in model:
                doc_embedding += model[word]
        doc_embedding = doc_embedding / len(document)
        data.append(doc_embedding)
    
    data = np.array(data)
    data = data.reshape((5000, dimensions))
    
    path = os.path.join(path_to_db, "fasttext")
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    with open(os.path.join(path, "documents.pickle"), "wb") as pickle_file:
        pickle.dump(documents2, pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def search_fasttext(req, path_to_db):
    path = os.path.join(path_to_db, "fasttext")
    
    with open(os.path.join(path, "documents.pickle"), "rb") as pickle_file:
        documents = pickle.load(pickle_file)
    
    with open(os.path.join(path, "data.pickle"), "rb") as pickle_file:
        data = pickle.load(pickle_file)
    
    model_file = './fasttext/model.model'
    model = KeyedVectors.load(model_file)
    
    dimensions = model.vector_size
    
    morph = pymorphy2.MorphAnalyzer()
    
    req = req.lower()
    req = re.split(r"[^а-яё]+", req)
    req = [morph.parse(word)[0].normal_form for word in req]
    
    req_embedding = np.array([0 for i in range(dimensions)], dtype="float64")
    for word in req:
        if word in model:
            req_embedding += model[word]
    req_embedding = req_embedding / len(req)
    
    result_dict = {" ".join(documents[i]): cosine(req_embedding, data[i]) for i in range(len(data))}
    output = sorted(result_dict.items(), key=lambda x: x[1])
    
    docs = [o[0] for o in output[:10]]
    scores = [o[1] for o in output[:10]]
    
    return docs, scores
