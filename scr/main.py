import os
import time
import logging
from create_index import create_index
from helper import unpack_tar_gz
from process import process_parent_dir
from re_extract import only_unextracted
from config import LOG_DIR, PARENT_DIR, EMB_DIR, INDEX_TO_UPLOAD

log_dir = LOG_DIR
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

parent_dir = PARENT_DIR
embedding_dir = EMB_DIR

#TODO: make an optional argument
password = os.getenv('ELASTIC_SEARCH_UNI_PASSWORD')
create_response = create_index(password, INDEX_TO_UPLOAD, 256, 1)

@only_unextracted(embedding_dir)
def main(unprocessed_archives):
    os.makedirs(embedding_dir, exist_ok=True)

    total_archives = len(unprocessed_archives)

    process_parent_dir(parent_dir)

    for idx, file_path in enumerate(unprocessed_archives, start=1):
        unpack_tar_gz(file_path, parent_dir)

        process_parent_dir(parent_dir)

        progress = (idx / total_archives) * 100 if total_archives else 100
        logging.info(f"Progress: {progress:.2f}%")

if __name__ == '__main__':
    main()
