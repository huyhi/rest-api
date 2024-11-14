import json
import os

from flask import Flask, request, Response
from flask import jsonify
from flask_cors import CORS, cross_origin

from service.chain import chat_streaming_output, summarize_output, literature_review_output, format_papers_in_prompt
from extension.ext import cached_data
from model.const import EMBED
from model.mongo import MongoQuerySchema
from service import mongo, lib
from service.mongo import query_docs

app = Flask(__name__, static_folder='./build', static_url_path='/')
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


def get_similarities_v2(paper_ids, embedding, dimensions, limit):
    if embedding not in EMBED.ALL:
        raise ValueError('Embedding must be one of {}'.format(EMBED.ALL))

    papers = mongo.query_doc_full_fields_by_ids(paper_ids)
    if not papers:
        return []

    if dimensions == "nD":
        return mongo.query_similar_doc_by_embedding_full(papers, embedding, limit)
    elif dimensions == '2D':
        return mongo.query_similar_doc_by_embedding_2d(papers, embedding, limit)


def get_similarities_by_abstract_v2(input_data, embedding, limit):
    return mongo.query_similar_doc_by_paper(input_data, embedding, limit)


@app.route('/getPapers', methods=['GET', 'POST'])
@cross_origin()
def get_papers():
    if request.method == 'GET':
        input_payload = request.args
    elif request.method == 'POST':
        input_payload = request.json

    query = MongoQuerySchema(
        title=input_payload.get('title'),
        abstract=input_payload.get('abstract'),
        author=input_payload.get('author'),
        source=input_payload.get('source'),
        keyword=input_payload.get('keyword'),
        min_year=input_payload.get('min_year'),
        max_year=input_payload.get('max_year'),
        min_citation_counts=input_payload.get('min_citation_counts'),
        max_citation_counts=input_payload.get('max_citation_counts'),
        id_list=input_payload.get('id_list'),
        offset=input_payload.get('offset', 0),
        limit=input_payload.get('limit', 1000)
    )
    return jsonify(query_docs(query))


@app.route('/getSimilarPapers', methods=['POST'])
@cross_origin()
def get_similar_papers():
    input_payload = request.json

    paper_ids = input_payload["input_data"]
    embedding = input_payload["embedding"]
    dimensions = input_payload["dimensions"]
    limit = int(input_payload["limit"])

    results = get_similarities_v2(paper_ids, embedding, dimensions, limit)
    if not results:
        return Response("[]", status=200, content_type='application/json')
    else:
        return Response(json.dumps(results), status=200, content_type='application/json')


@app.route('/getSimilarPapersByKeyword', methods=['POST'])
@cross_origin()
def get_similar_papers_by_keyword():
    return Response('API NOT IN USE RIGHT NOW', status=200, content_type='application/json')


@app.route('/getSimilarPapersByAbstract', methods=['POST'])
@cross_origin()
def get_similar_papers_by_abstract():
    input_payload = request.json
    limit = int(input_payload["limit"])
    embedding = input_payload["embedding"]
    input_data = input_payload["input_data"]

    results = get_similarities_by_abstract_v2(input_data, embedding, limit)
    if results is None:
        return Response("[]", status=200, content_type='application/json')
    else:
        # apply the original limit to the final df
        return Response(json.dumps(results), status=200, content_type='application/json')


@app.route('/checkoutPapers', methods=['POST'])
@cross_origin()
def checkout_papers():
    input_payload = request.json
    paper_ids = input_payload["input_data"]

    filename = "papers-checked-out.bibtex"
    papers = mongo.query_doc_by_ids(paper_ids)

    response_text = '\n'.join([lib.bib_template(paper) for paper in papers])
    return Response(response_text, mimetype="text/plain", headers={"Content-Disposition": "attachment;" + filename})


@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    text = request.json.get('text', '')
    chat_history = request.json.get('chatHistory', [])
    if not text:
        return Response(tuple('Please Input Your Text'))
    if len(text) > 1e6:
        return Response(tuple('Too Long Text'))
    return Response(chat_streaming_output(text, chat_history), status=200, content_type='text/plain')


@app.route('/getUmapPoints', methods=['GET'])
@cross_origin()
def get_umap_points():
    return jsonify(cached_data.get_umap_points())


@app.route('/getMetaData', methods=['GET'])
@cross_origin()
def get_metas():
    authors = mongo.get_distinct_authors()  # Retrieve distinct authors from the database
    sources = mongo.get_distinct_sources()  # Retrieve distinct sources from the database
    keywords = mongo.get_distinct_keywords()  # Retrieve distinct keywords from the database
    years = mongo.get_distinct_years()  # Retrieve distinct years from the database
    titles = mongo.get_distinct_titles()  # Retrieve distinct titles from the database
    citation_counts = mongo.get_distinct_citation_counts()

    return jsonify({
        'authors': authors,
        'sources': sources,
        'keywords': keywords,
        'years':years,
        'titles':titles,
        'citation_counts': citation_counts
    })


@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize():
    prompt = request.json.get('prompt', '')
    paper_ids = request.json.get('ids', [])
    if not paper_ids:
        return Response(tuple('Saved paper list is empty'), status=200, content_type='application/json')

    selected_papers = mongo.query_doc_by_ids(paper_ids)

    return Response(summarize_output({
        'prompt': prompt,
        'content': format_papers_in_prompt(selected_papers)
    }), status=200, content_type='application/json')


@app.route('/literatureReview', methods=['POST'])
@cross_origin()
def literature_review():
    paper_ids = request.json.get('ids', [])
    prompt = request.json.get('prompt', '')
    if not paper_ids:
        return Response(tuple('Saved paper list is empty'), status=200, content_type='application/json')

    selected_papers = mongo.query_doc_by_ids(paper_ids)

    return Response(literature_review_output({
        'prompt': prompt,
        'content': format_papers_in_prompt(selected_papers)
    }), status=200, content_type='application/json')



@app.route('/')
@cross_origin()
def index():
    return app.send_static_file('index.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    cached_data.init()
    app.run(host='0.0.0.0', port=port)  # Run it, baby!
