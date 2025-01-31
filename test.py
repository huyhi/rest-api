from pprint import pprint

from service.mongo import docs_collection

all_fields_doc = docs_collection.find_one({"ID": "985"})
pprint(all_fields_doc)

