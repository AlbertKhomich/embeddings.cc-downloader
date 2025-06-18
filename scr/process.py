import logging
import torch
import shutil
import os
from helper import get_file_paths
from prepare_data import post_embeddings

password = os.getenv('ELASTIC_SEARCH_UNI_PASSWORD')
index_name = 'whale'

def process_parent_dir(parent_dir):
    for e in os.listdir(parent_dir):
        e_path = os.path.join(parent_dir, e)
        process_emb_dir(e_path)

def cleanup(dir_path):
    shutil.rmtree(dir_path)
    logging.info(f"Cleaned up {dir_path}")

def process_emb_dir(embedding_dir):
    logging.info('Uploading started.')

    file_paths = get_file_paths(embedding_dir)

    logging.info(f"File paths: {file_paths}")

    model_path = file_paths.get('model')
    if not model_path:
        logging.error(f"No model in: {embedding_dir}")
        cleanup(embedding_dir)
        return

    try:
        ckpt = torch.load(model_path, map_location='cpu')
    except Exception as e:
        logging.error(f"Failed to load model at {model_path}: {e}")
        cleanup(embedding_dir)
        return

    state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    entity_keys = [k for k in state_dict.keys() if 'entity_embeddings.weight' in k]
    if not entity_keys:
        logging.error(f"No entity_embeddings.weight key in {model_path}")
        cleanup(embedding_dir)
        return
    if len(entity_keys) > 1:
        logging.warning(f"Multiple matches for entity_embeddings.weight: {entity_keys}; using the first one.")
    emb_key = entity_keys[0]
    logging.info(f"Using embedding key: {emb_key}")

    entity_idx_path = file_paths.get('entity_to_idx.p') or file_paths.get('entity_to_idx.csv')
    if not entity_idx_path:
        logging.error(f"No entity index file in {embedding_dir}")
        cleanup(embedding_dir)
        return

    try:
        add_response_e = post_embeddings(
            state_dict, 
            entity_idx_path,
            emb_key,
            password, 
            index_name
        )
        logging.info(f"Uploading entities: {add_response_e}")
    except Exception as e:
        logging.error(f"Error adding entity embeddings from {entity_idx_path}: {e}")
        cleanup(embedding_dir)
        return

    # relation_idx_path = file_paths.get('relation_to_idx.p') or file_paths.get('relation_to_idx.csv')
    # if relation_idx_path:
    #     try:
    #         add_response_r = post_embeddings(model, relation_idx_path, 'relation_embeddings.weight', password, index_name)
    #         logging.info(f"Uploading relations: {add_response_r}")
    #     except Exception as e:
    #             logging.error(f"Error adding relation embeddings from {relation_idx_path}: {e}")
    # else:
    #     logging.warning(f"No relation index file found in {embedding_dir}; skipping relation upload.")

    shutil.rmtree(embedding_dir)
    logging.info('Directory finished.')