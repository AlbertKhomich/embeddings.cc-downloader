import pickle
import csv
import os
import logging
from add_data import add_data
from helper import chunk_docs

def load_mapping_pickle(mapping_path):
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def load_mapping_csv(mapping_path):
    mapping = {}
    with open(mapping_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                idx = int(row[0])
            except ValueError:
                continue
            key = row[1].strip()
            mapping[key] = idx
    return mapping

def extract_embeddings(model, mapping_path, embedding_key):
    ext = os.path.splitext(mapping_path)[1].lower()
    if ext == '.p':
        mapping = load_mapping_pickle(mapping_path)
    elif ext == '.csv':
        mapping = load_mapping_csv(mapping_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    idx_to_val = {v: k for k, v in mapping.items()}

    if embedding_key not in model:
        raise KeyError(f"Embeddings with key '{embedding_key}' not found in model. Check model keys: {list(model.keys())}")

    embeddings = model[embedding_key].detach().numpy()

    docs = []
    for i in range(len(embeddings)):
        entity = idx_to_val[i].strip('<>')
        doc = [entity, embeddings[i].tolist()]
        docs.append(doc)

    return docs    

def post_embeddings(model, idx_path, embeddings_weight, password, index_name):
    docs = extract_embeddings(model, idx_path, embeddings_weight)
    logging.info(f'Test vector: {docs[0]}')
    max_payload_size = 1024 * 1024 # 1 MB

    responses = []
    logging.info("Transfering...")

    for chunk in chunk_docs(docs, max_payload_size):        
        try:
            response = add_data(password, index_name, chunk)
        except Exception as e:
            logging.info(f"Failed to add data after retries: {e}")
            response = None
        responses.append(response)

    return responses
