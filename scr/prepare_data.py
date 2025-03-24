import pickle
import json

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
