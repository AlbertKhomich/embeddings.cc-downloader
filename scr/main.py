import os
import csv
import json
import time
import torch
import logging
import shutil
from elasticsearch import Elasticsearch
from create_index import create_index
from prepare_data import post_embeddings
from helper import get_file_paths, unpack_tar_gz, clean_dir
from re_extract import only_unextracted

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
embedding_dir = '/scratch/hpc-prf-whale/WHALE-output/embeddings/hreview/models'
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

@only_unextracted(embedding_dir, '/scratch/hpc-prf-whale/albert/uploader_embeddings/logs/hreview_checkpoint.log')
def main(unprocessed_archives):
    os.makedirs(embedding_dir, exist_ok=True)

    total_archives = len(unprocessed_archives)

    for e in os.listdir(parent_dir):
        e_path = os.path.join(parent_dir, e)
        process_emb_dir(e_path)

    for idx, file_path in enumerate(unprocessed_archives, start=1):
        unpack_tar_gz(file_path, parent_dir)

        for e in os.listdir(parent_dir):
            e_path = os.path.join(parent_dir, e)
            process_emb_dir(e_path)

        progress = (idx / total_archives) * 100 if total_archives else 100
        logging.info(f"Progress: {progress:.2f}%")

if __name__ == '__main__':
    main()
