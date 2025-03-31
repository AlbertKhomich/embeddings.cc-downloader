import os
import csv
import json
import time
import torch
import logging
from elasticsearch import Elasticsearch
from create_index import create_index
from prepare_data import post_embeddings
from helper import get_file_paths, unpack_tar_gz, clean_dir

log_dir = '/scratch/hpc-prf-whale/albert/uploader_embeddings/logs'
log_filename = os.path.join(log_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '_log.log')
error_log_filename = os.path.join(log_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '_error.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

error_handler = logging.FileHandler(error_log_filename)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.getLogger().addHandler(error_handler)

password = os.getenv('ELASTIC_SEARCH_UNI_PASSWORD')
index_name = 'whale'
parent_dir = '/scratch/hpc-prf-whale/albert/uploader_embeddings/data'
embedding_dir = '/scratch/hpc-prf-whale/WHALE-output/embeddings/geo/models'
# dimensions = 256
# create_response = create_index(password, index_name, dimensions)
# print("Index creation response:", create_response)

def process_emb_dir(embedding_dir):
    logging.info('Downloading started.')

    file_paths = get_file_paths(embedding_dir)

    logging.info(f"File paths: {file_paths}")

    entity_idx_path = file_paths['entity_to_idx.p']
    relation_idx_path = file_paths['relation_to_idx.p']
    model_path = file_paths['model.pt']

    try:
        logging.info("Preparing data for tranfering.")
        model = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        logging.error(f"No model in: {model_path}")
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

    logging.info('Directory finished.')

def main():
    os.makedirs(embedding_dir, exist_ok=True)

    for filename in os.listdir(embedding_dir):
        if filename.endswith('.tar.gz'):
            file_path = os.path.join(embedding_dir, filename)

            unpack_tar_gz(file_path, parent_dir)

            for e in os.listdir(parent_dir):
                e_path = os.path.join(parent_dir, e)
                process_emb_dir(e_path)

            clean_dir(parent_dir)

main()
