import pickle
import json
import logging
from add_data import add_data
from helper import chunk_docs

def extract_embeddings(model, mapping_path, embedding_key):
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
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
