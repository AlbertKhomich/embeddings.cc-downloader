import torch
import json
import csv
from elasticsearch import Elasticsearch
from create_index import create_index
from add_data import add_data
from prepare_data import extract_embeddings

entity_idx_path = '../data/Keci_GPU_wikizero.org.txt/entity_to_idx.p'
relation_idx_path = '../data/Keci_GPU_wikizero.org.txt/relation_to_idx.p'
model_path = '../data/Keci_GPU_wikizero.org.txt/model.pt'
entity_json = '../data/Keci_GPU_wikizero.org.txt/entity_embeddings.json'
relation_json = '../data/Keci_GPU_wikizero.org.txt/relation_embeddings.json'

password = "="
index_name = 'whale'
dimensions = 256
# create_response = create_index(password, index_name, dimensions)
# print("Index creation response:", create_response)

model = torch.load(model_path, map_location=torch.device('cpu'))

with open(relation_json, 'r') as file:
    docs = json.load(file)

docs_e = extract_embeddings(model, entity_idx_path, 'entity_embeddings.weight')
add_response_e = add_data(password, index_name, docs)
print("Add insert response:", add_response_e)

# docs_r = extract_embeddings(model, relation_idx_path, 'relation_embeddings.weight')
# add_response_r = add_data(password, index_name, docs_e)
# print("Add insert response:", add_response_r)
# print(docs_r[1])
