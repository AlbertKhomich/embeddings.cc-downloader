import os
import shutil
import tarfile
import logging
import json

def get_file_paths(dir):
    filenames = [
        'entity_to_idx.p',
        'relation_to_idx.p',
        'model.pt',
        'model_partial_0.pt',
        'entity_to_idx.csv',
        'relation_to_idx.csv'
    ]

    file_paths = {}

    for filename in filenames:
        file_path = os.path.join(dir, filename)

        if os.path.exists(file_path):
            file_paths[filename] = file_path

    return file_paths

def unpack_tar_gz(archive_path, target_dir):
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)
            logging.info(f"Successfully extracted {archive_path} to {target_dir}")
    except Exception as e:
        logging.error(f"Error extracting {archive_path}: {e}")

def clean_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)

            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        logging.info(f"Contents of '{path}' have been deleted.")
    else:
        logging.error(f"The path '{path}' does not exist or is not a directory.")

def chunk_docs(docs, max_payload_size):
    chunk = []
    current_size = 0
    for doc in docs:
        doc_json = json.dumps(doc)
        doc_size = len(doc_json.encode('utf-8'))

        if current_size + doc_size > max_payload_size:
            yield chunk
            chunk = [doc]
            current_size = doc_size
        else:
            chunk.append(doc)
            current_size += doc_size
    if chunk:
        yield chunk