import json
import os
import sys
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
import pymongo
import requests
from flask import Flask, request, Response
from flask import jsonify
from flask_cors import CORS, cross_origin
from langchain_openai import OpenAIEmbeddings
from sklearn.preprocessing import MinMaxScaler

import config
from service.chain import chat_streaming_output, summarize_output, literature_review_output
from extension.ext import cached_data
from model.const import EMBED
from model.mongo import MongoQuerySchema
from service.mongo import query_docs

client = None
docs = None
scaler = MinMaxScaler(feature_range=(0,1))
app = Flask(__name__, static_folder='./build', static_url_path='/')
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
df = None

ada_embedding_func = OpenAIEmbeddings(model='text-embedding-ada-002')

restricted_column_list = ['glove_embedding', 'specter_embedding', 'ada_embedding']
query_index = {
    "glove_embedding": None,
    "specter_embedding": None,
    "ada_embedding": None
}


def create_query_index():
    global df, query_index

    # Glove
    if "glove_embedding" in df.columns:
        null_free_df = df[df['glove_embedding'].str.len() > 0].copy()
        null_free_df.dropna(subset=['glove_embedding'], inplace=True)
        xb = (np.array(null_free_df["glove_embedding"].tolist())).astype('float32')
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print("created query index for glove")
        query_index['glove_embedding'] = index

    # Specter
    if "specter_embedding" in df.columns:
        null_free_df = df[df['specter_embedding'].str.len() > 0].copy()
        null_free_df.dropna(subset=['specter_embedding'], inplace=True)
        xb = (np.array(null_free_df["specter_embedding"].tolist())).astype('float32')
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print("created query index for specter")
        query_index['specter_embedding'] = index

    # Ada
    if "ada_embedding" in df.columns:
        null_free_df = df[df['ada_embedding'].str.len() > 0].copy()
        null_free_df.dropna(subset=['ada_embedding'], inplace=True)
        xb = (np.array(null_free_df["ada_embedding"].tolist())).astype('float32')
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print("created query index for ada")
        query_index['ada_embedding'] = index


def load_data():
    global client, docs, df

    if config.data_source == "json": # Reading from LOCAL FILE

        json_data = pd.read_json(config.raw_json_datafile, orient="records")
        df_data = pd.DataFrame(json_data)
        df_data.dropna(subset=['Title', "Authors"], inplace=True)

    elif config.data_source == "mongodb": # Reading from MONGODB

        # add in <user> and <pwd>
        client = pymongo.MongoClient(config.mongodb_connection_uri)
        # Choose the database
        db = client[config.mongodb_database]
        # Get the document collection
        docs = db[config.mongodb_collection]
        # print('There are ' + str(docs.count_documents({})) + " papers to load. Loading them now...")

        # db query to restrict sources
        query = {
            "Source": {"$not":{"$in":[]}},
            "Title": {"$not": {"$eq": None}},
            "Authors": {"$not": {"$eq": None}}
        }

        # db query
        query = {"Title": {"$not": {"$eq": None}}, "Authors": {"$not": {"$eq": None}}}

        # db projection
        # Uncomment if ALL but specter_ or glove_ related attributes needed
        projection = {"specter_embedding": 0, "specter_umap": 0}
        projection = {"glove_embedding": 0, "glove_umap": 0}
        df_data = pd.DataFrame(list(docs.find(query, projection)))

        # Uncomment if both glove and specter are needed
        df_data = pd.DataFrame(list(docs.find(query)))

        # Drop all indices
        docs.drop_indexes()

        # 2D Indices
        if "glove_umap" in df_data.columns:
            docs.create_index([("glove_umap", "2d")])
        if "specter_umap" in df_data.columns:
            docs.create_index([("specter_umap", "2d")])

    else:
        print("invalid config.data_source. Valid values are 'json', 'mongodb'")
        sys.exit()

    # Common to either formats
    if "glove_embedding" in df_data.columns:
        df_data['glove_embedding'] = df_data['glove_embedding'].apply(lambda x: np.array(x))
    if "specter_embedding" in df_data.columns:
        df_data['specter_embedding'] = df_data['specter_embedding'].apply(lambda x: np.array(x))
    if "ada_embedding" in df_data.columns:
        df_data['ada_embedding'] = df_data['ada_embedding'].apply(lambda x: np.array(x))

    if "glove_umap" in df_data.columns:
        df_data['glove_umap'] = df_data['glove_umap'].apply(lambda x: np.array(x))
    if "ada_umap" in df_data.columns:
        df_data['ada_umap'] = df_data['ada_umap'].apply(lambda x: np.array(x))
    if "specter_umap" in df_data.columns:
        df_data['specter_umap'] = df_data['specter_umap'].apply(lambda x: np.array(x))
    if "specter_umap" in df_data.columns:
        df_data['Authors'] = df_data['Authors'].apply(lambda x: np.array(x))
    if "specter_umap" in df_data.columns:
        df_data['Keywords'] = df_data['Keywords'].apply(lambda x: np.array(set(x)) if isinstance(x, list) else np.array(x))

    print("loaded data with ", len(df_data), "records")
    df = df_data

    # Taking a MongoDB BACKUP
    # cursor = docs.find(query)
    # with open('data/VitaLITy-MongoDB-Collection-1.0.0.json', 'w') as file:
    #     file.write('[')
    #     for document in cursor:
    #         file.write(dumps(document))
    #         file.write(',')
    #     file.write(']')

    # # Taking a LOCAL BACKUP
    # # df.to_json("data/VitaLITy-1.0.0.json", orient="records", default_handler=str)


def get_query_vector(input_vector_indices, embedding, df):
    try:
        input_vectors = df[df.index.isin(input_vector_indices)][embedding]
        input_vector = np.array(np.mean(input_vectors)).astype(('float32'))
        return np.array([input_vector])
    except Exception as e:
        print(e)
        print("unable to find query vector")
        return None


def get_similarities(df, paper_ids, embedding, dimensions, limit):
    global query_index, docs
    if dimensions == "nD":
        embedding_col = embedding + "_embedding"  # can be specter_embedding, glove_embedding

        input_vector_indices = df.index[df['ID'].isin(paper_ids)]  # lookup indices of the papers to query
        query_vector = get_query_vector(input_vector_indices, embedding_col, df)  # create query_vector

        # It can be for *some* specter_embeddings since the API may not have worked.
        if query_vector is not None:
            try:
                squared_distances, indices = query_index[embedding_col].search(query_vector, limit)  # lookup similar papers in the query_index

                # Find the papers at those respective indices
                null_free_df = df[df[embedding_col].str.len() > 0].copy()
                null_free_df.dropna(subset=[embedding_col], inplace=True)
                df_similar = null_free_df.iloc[indices.tolist()[0]].copy()

                similarities = np.reciprocal(squared_distances.tolist())  # similarity = reciprocal of distance
                temp_sim = np.copy(similarities)
                temp_sim[temp_sim == np.inf] = np.min(temp_sim)  # Replace all infinite values to the min of the array.
                similarities[similarities == np.inf] = np.max(temp_sim) + 1  # Use the now second largest value as my max distance since scaling doesn't like infinity
                # similarities_sorted = -np.sort(-similarities)  # sort in descending order

                # scale similarities between 0,1
                scaled_similarities = scaler.fit_transform(similarities.reshape(-1,1))

                df_similar.loc[:,"Distance"] = squared_distances.tolist()[0]   # add new column with Distance
                df_similar.loc[:,"Sim"] = [round(s[0], 4) for s in scaled_similarities]  # add new column with similarity scores and scale it to [0, 1] range
                df_similar.loc[:,"Sim_Rank"] = range(1, len(df_similar)+1)  # add new column with the similarity ranking
                df_similar.loc[:,"Distance"] = [round(d, 2) for d in df_similar["Distance"].values]  # Distance column

                # Remove papers that belonged to the input paper list
                df_similar_filtered = df_similar[~df_similar["ID"].isin(list(paper_ids))].copy()

                return df_similar_filtered.loc[:, ~df_similar_filtered.columns.isin(restricted_column_list)]

            except Exception as e:
                return None
        else:
            return None

    elif dimensions == "2D":
        embedding_col = embedding + "_umap"  # can be specter_umap, glove_umap

        df_filtered = df[df['ID'].isin(paper_ids)]
        try:
            coords_ndarray = np.array(df_filtered[[embedding_col]].values.tolist())
            coords = np.mean(coords_ndarray, axis=0)

            # Using the $near API
            # df_similar = pd.DataFrame(list(docs.find({embedding_col:{"$near":list(coords[0])}}).limit(limit)))

            # Using the $geoNear API
            similarity_output = docs.aggregate([{
                "$geoNear":{
                    "key": embedding_col,
                    "near": list(coords[0]),
                    # "near": { "type": "Point", "coordinates": list(coords[0]) },
                    "distanceField": "Distance",
                }
            }])

            counter = 0
            similar_documents = []
            for output in similarity_output:
                similar_documents.append(output)
                counter += 1
                if counter == limit:
                    break

            df_similar = pd.DataFrame(similar_documents)

            similarities = np.reciprocal(df_similar["Distance"].tolist())  # similarity = reciprocal of distance
            temp_sim = np.copy(similarities)
            temp_sim[temp_sim == np.inf] = np.min(temp_sim)  # Replace all infinite values to the min of the array.
            similarities[similarities == np.inf] = np.max(temp_sim) + 1  # Use the now second largest value as my max distance since scaling doesn't like infinity
            # similarities_sorted = -np.sort(-similarities)  # sort in descending order

            # scale similarities between 0,1
            scaled_similarities = scaler.fit_transform(similarities.reshape(-1, 1))

            df_similar.loc[:,"Sim"] = [round(s[0], 4) for s in scaled_similarities]  # add new column with similarity scores and scale it to [0, 1] range
            df_similar.loc[:,"Sim_Rank"] = range(1, len(df_similar)+1)  # add new column with the similarity ranking
            df_similar.loc[:,"Distance"] = [round(d, 2) for d in df_similar["Distance"].values]  # Distance column

            # Remove papers that belonged to the input paper list
            df_similar_filtered = df_similar[~df_similar["ID"].isin(paper_ids)].copy()

            return df_similar_filtered.loc[:, ~df_similar_filtered.columns.isin(restricted_column_list)]

        except Exception:
            return None
    else:
        return None


def is_keyword_match(paper_keywords, keywords_to_match):
    if paper_keywords is None:
        return False

    if bool(set([a.lower() for a  in paper_keywords]) & set([b.lower() for b in keywords_to_match])):
        return True

    return False


def get_similarities_by_keyword(df, keywords_to_match, limit):
    try:
        return df[df['Keywords'].apply(lambda paper_keywords: is_keyword_match(paper_keywords, keywords_to_match))].head(limit)
    except Exception:
        return None


ABSTRACT_SIMILARITY_REST_API_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16
def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(ABSTRACT_SIMILARITY_REST_API_URL, json=chunk)
        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")
        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]
    return embeddings_by_paper_id


def get_similarities_by_abstract(df, embedding, input_data, limit):
    try:
        if embedding == EMBED.ADA:
            query_vector = ada_embedding(input_data)
            embedding_col = 'ada_embedding'
        else:
            query_vector = specter_embedding(input_data)
            embedding_col = 'specter_embedding'

        # It can be for *some* specter_embeddings since the API may not have worked.
        if query_vector is not None and query_vector.shape[1] != 0:
            squared_distances, indices = query_index[embedding_col].search(query_vector, limit)  # lookup similar papers in the query_index

            null_free_df = df.copy()
            null_free_df.dropna(subset=[embedding_col], inplace=True)
            df_similar = null_free_df.iloc[indices.tolist()[0]].copy()

            similarities = np.reciprocal(squared_distances.tolist())  # similarity = reciprocal of distance
            temp_sim = np.copy(similarities)
            temp_sim[temp_sim == np.inf] = np.min(temp_sim)  # Replace all infinite values to the min of the array.
            similarities[similarities == np.inf] = np.max(temp_sim) + 1  # Use the now second largest value as my max distance since scaling doesn't like infinity
            # similarities_sorted = -np.sort(-similarities)  # sort in descending order

            # scale similarities between 0,1
            scaled_similarities = scaler.fit_transform(similarities.reshape(-1,1))

            df_similar.loc[:,"Distance"] = squared_distances.tolist()[0]   # add new column with Distance
            df_similar.loc[:,"Sim"] = [round(s[0], 4) for s in scaled_similarities]  # add new column with similarity scores and scale it to [0, 1] range
            df_similar.loc[:,"Sim_Rank"] = range(1, len(df_similar)+1)  # add new column with the similarity ranking
            df_similar.loc[:,"Distance"] = [round(d, 2) for d in df_similar["Distance"].values]  # Distance column

            return df_similar.loc[:, ~df_similar.columns.isin(restricted_column_list)]
        else:
            return None
    except Exception as e:
        return None


def specter_embedding(input_data) -> np.array:
    payload = [{
        "paper_id": "sample_id",
        "title": input_data['title'],
        "abstract": input_data['abstract']
    }]
    embeddings = embed(payload)
    return np.array([embeddings['sample_id']]).astype('float32')


def ada_embedding(input_data) -> np.array:
    embeddings = ada_embedding_func.embed_query(input_data['abstract'])
    return np.array([embeddings]).astype('float32')



@app.route('/getPapers', methods = ['GET'])
@cross_origin()
def get_papers():
    # global df
    # df_of_interest = df.loc[:, ~df.columns.isin(restricted_column_list)]
    # response = df_of_interest.to_json(orient="records", default_handler=str)
    # return Response(response, status=200, content_type='application/json')

    query = MongoQuerySchema(limit=1000)
    return jsonify(query_docs(query))



@app.route('/getSimilarPapers', methods=['POST'])
@cross_origin()
def get_similar_papers():
    global df
    input_payload = request.json
    input_type = input_payload["input_type"]
    if input_type == "Title":
        paper_titles = input_payload["input_data"]
        paper_ids = list(df[df["Title"].isin(paper_titles)]["ID"])
    elif input_type == "ID":
        paper_ids = input_payload["input_data"]
    else:
        return Response("Valid input_type = Title / ID")

    embedding = input_payload["embedding"]
    dimensions = input_payload["dimensions"]
    limit = int(input_payload["limit"])

    # send limit*2 as the desired limit so that if there are input papers in the output papers, then we have to discard them; and the extra limit will help ensure the final limit stays right
    results_df = get_similarities(df, paper_ids, embedding, dimensions, limit * 2)
    if results_df is None:
        return Response("[]", status=200, content_type='application/json')
    else:
        # apply the original limit to the final df
        return Response(results_df.head(limit).to_json(orient="records", default_handler=str), status=200, content_type='application/json')


@app.route('/getSimilarPapersByKeyword', methods=['POST'])
@cross_origin()
def get_similar_papers_by_keyword():
    global df
    input_payload = request.json
    keywords = input_payload["input_data"]
    limit = int(input_payload["limit"])
    results_df = get_similarities_by_keyword(df, keywords, limit)
    if results_df is None:
        return Response("[]", status=200, content_type='application/json')
    else:
        return Response(results_df.to_json(orient="records", default_handler=str), status=200, content_type='application/json')



@app.route('/getSimilarPapersByAbstract', methods=['POST'])
@cross_origin()
def get_similar_papers_by_abstract():
    global df
    input_payload = request.json
    limit = int(input_payload["limit"])
    results_df = get_similarities_by_abstract(
        df, input_payload["embedding"], input_payload["input_data"], limit
    )
    if results_df is None:
        return Response("[]", status=200, content_type='application/json')
    else:
        # apply the original limit to the final df
        return Response(results_df.to_json(orient="records", default_handler=str), status=200, content_type='application/json')


@app.route('/checkoutPapers', methods=['POST'])
@cross_origin()
def checkout_papers():
    global df
    input_payload = request.json
    input_type = input_payload["input_type"]
    if input_type == "Title":
        paper_titles = input_payload["input_data"]
        paper_ids = list(df[df["Title"].isin(paper_titles)]["ID"])
    elif input_type == "ID":
        paper_ids = input_payload["input_data"]
    else:
        return Response("Valid input_type = Title / ID")

    # df_checkout = df.loc[:, ~df.columns.isin(["_id", "glove_embedding", "specter_embedding", "glove_umap", "specter_umap", "AbstractLength", "glove_kmeans_cluster", "specter_kmeans"])]
    df_checkout = df.loc[:, df.columns.isin(["ID","Title","Authors","Source","Year"])]
    papers = json.loads(df_checkout[df_checkout["ID"].isin(paper_ids)].to_json(orient="records", default_handler=str))
    filename="papers-checked-out.bibtex"

    response_text = '\n'.join([bib_template(paper) for paper in papers])
    return Response(response_text, mimetype="text/plain", headers={"Content-Disposition": "attachment;" + filename})


def bib_template(paper):
    title = paper.get('Title', '')
    year = paper.get('Year', '')
    authors = []
    for author in paper.get('Authors', []):
        name = author.split(' ')
        authors.append(f'{name[1]}, {name[0]}')
    bib_id = f"{authors[0].split(',')[0]}{year}{title.split(' ')[0]}"

    return f'''
@article{{{bib_id},
  title={{{title}}},
  author={{{' and '.join(authors)}}},
  year={{{year}}}
}}
'''



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


def format_papers_in_prompt(papers):
    return '\n'.join([(f" --- "
            f"Title: {paper.get('Title', '')}\n"
            f"Authors: {paper.get('Authors', '')}\n"
            f"Abstract: {paper.get('Abstract', '')}\n"
            f"Source: {paper.get('Source', '')}\n"
            f"Year: {paper.get('Year', '')}\n"
            f"Keywords: {paper.get('Keywords', '')}\n"
            f" --- ") for paper in papers])


@app.route('/getUmapPoints', methods=['GET'])
@cross_origin()
def get_umap_points():
    return jsonify(cached_data.get_umap_points())


@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize():
    prompt = request.json.get('prompt', '')
    paper_ids = request.json.get('ids', [])
    if not paper_ids:
        return Response(tuple('Saved paper list is empty'), status=200, content_type='application/json')

    global df
    df_summarize = df.loc[:, df.columns.isin(["ID", "Title", "Authors", "Abstract", "Source", "Year", "Keywords"])]
    selected_papers = json.loads(df_summarize[df_summarize["ID"].isin(paper_ids)].to_json(orient="records", default_handler=str))

    return Response(summarize_output({
        'prompt': prompt,
        'content': format_papers_in_prompt(selected_papers)
    }), status=200, content_type='application/json')


@app.route('/literatureReview', methods=['POST'])
@cross_origin()
def literatureReview():
    paper_ids = request.json.get('ids', [])
    prompt = request.json.get('prompt', '')
    if not paper_ids:
        return Response(tuple('Saved paper list is empty'), status=200, content_type='application/json')

    global df
    df_summarize = df.loc[:, df.columns.isin(["ID", "Title", "Authors", "Abstract", "Source", "Year", "Keywords"])]
    selected_papers = json.loads(df_summarize[df_summarize["ID"].isin(paper_ids)].to_json(orient="records", default_handler=str))

    return Response(literature_review_output({
        'prompt': prompt,
        'content': format_papers_in_prompt(selected_papers)
    }), status=200, content_type='application/json')


@app.route('/')
@cross_origin()
def index():
    return app.send_static_file('index.html')


def load_data_and_create_index():
    load_data()
    create_query_index()


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))

    # This thread was added so that Heroku can run Flask and bind the $PORT ASAP. If it is unable to do that within 30 seconds, it will timeout.
    # Our process takes much longer than that so we start a new thread for it.
    # Thread(target=load_data_and_create_index).start()
    load_data_and_create_index()

    app.run(host='0.0.0.0', port=port)  # Run it, baby!