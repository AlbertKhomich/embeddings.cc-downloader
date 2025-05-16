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

def process_emb_dir(embedding_dir):
    logging.info('Downloading started.')

    file_paths = get_file_paths(embedding_dir)

    logging.info(f"File paths: {file_paths}")

    entity_idx_path = file_paths.get('entity_to_idx.p') or file_paths.get('entity_to_idx.csv')
    relation_idx_path = file_paths.get('relation_to_idx.p') or file_paths.get('relation_to_idx.csv')
    model_path = file_paths.get('model.pt') or file_paths.get('model_partial_0.pt')

    try:
        logging.info("Preparing data for transfering.")
        model = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        logging.error(f"No model in: {model_path}")
        shutil.rmtree(embedding_dir)
        return

    try:
        add_response_e = post_embeddings(model, entity_idx_path, 'entity_embeddings.weight', password, index_name)
        logging.info(f"Uploading entities: {add_response_e}")
    except Exception as e:
        logging.error(f"Error adding entity embeddings: {e}")

    try:
        add_response_r = post_embeddings(model, relation_idx_path, 'relation_embeddings.weight', password, index_name)
        logging.info(f"Uploading relations: {add_response_r}")
    except Exception as e:
            logging.error(f"Error adding relation embeddings: {e}")

    shutil.rmtree(embedding_dir)
    logging.info('Directory finished.')