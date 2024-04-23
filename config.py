data_source = "json" # 'json' or 'mongodb' 

# if `data_source` = 'json'
# Note: 2-D search only works with MongoDB and not a raw JSON file.
# raw_json_datafile = 'data/VitaLITy-1.0.0.json'
raw_json_datafile = 'data/VitaLITy-1.1.0.json'

# if `data_source` = 'mongodb'
mongodb_connection_uri = None
mongodb_database = 'vissimilarity'
mongodb_collection = 'docs'

DB_FOLDER_NAME = 'data.db'
COLLECTION_NAME = 'paper_v3'  # v3 -> Vis and ada embedding
