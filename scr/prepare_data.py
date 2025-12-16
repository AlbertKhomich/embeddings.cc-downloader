import pickle
import csv
import time
import sys
import os
import logging
from add_data import add_data
from helper import chunk_docs
from config import BATCH_SIZE

csv.field_size_limit(sys.maxsize)

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

    for i, emb_tensor in enumerate(model[embedding_key]):
        entity = idx_to_val[i].strip('<>')

        #TODO: remove this below!
        if entity.startswith("http://whale.data.dice-research.org/.well-known/genid/"):
            continue

        vec = emb_tensor.detach().cpu().numpy().tolist()
        yield [entity, vec]

def post_embeddings(model, idx_path, embeddings_weight, password, index_name):    
    docs = extract_embeddings(model, idx_path, embeddings_weight)
    max_payload_size = BATCH_SIZE

    responses = []
    logging.info("Transfering...")

    for i, chunk in enumerate(chunk_docs(docs, max_payload_size), start=1):
        try:
            response = add_data(password, index_name, chunk)
        except Exception as e:
            logging.exception(f"[Chunk #{i}] Failed to add data after retries")
            
            response = None
        responses.append(response)

        if i % 10 == 0:
            logging.info(f"Completed {i} bulks - sleeping to let merges catch up")
            time.sleep(2)

    return responses
