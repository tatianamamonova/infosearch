import logging
import os
from flask import Flask
from flask import url_for, render_template, request, redirect
from models import *


logging.basicConfig(filename="search.log", format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.DEBUG)
logging.info("Program started")

db_path = "data"

app = Flask(__name__)

@app.route('/search')
def printsearch(query, docs, scores):
    logging.info("Search started")
    return render_template('result.html', query=query, docs=docs, n=len(docs), scores=scores)

@app.route('/')
def index():
    logging.info("Opening the main page")
    if request.args:
        query = request.args["query"]
        if request.args["method"] == "FastText":
            docs, scores = search_fasttext(query, db_path)
            return printsearch(query, docs, scores)
        elif request.args["method"] == "BM25":
            docs, scores = search_bm25(query, db_path)
            return printsearch(query, docs, scores)
        else:
            docs, scores = search_tfidf(query, db_path)
            return printsearch(query, docs, scores)
    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists(db_path):
        os.mkdir(db_path)
        create_db_tfidf("quora.csv", db_path)
        create_db_bm25("quora.csv", db_path)
        create_db_fasttext("quora.csv", db_path)
    app.run(debug=True)
