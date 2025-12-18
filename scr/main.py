import os
import time
import logging
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-index",
        action='store_true',
        help='Create/unsure Elasticsearch index before processing',
    )

    parser.add_argument('--index', default=INDEX_TO_UPLOAD, help="Elasticsearch index name")
    parser.add_argument("--dims", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--shards", type=int, default=1, help="Number of shards")

    return parser.parse_args()

args = parse_args()

@only_unextracted(embedding_dir)
def main(unprocessed_archives: list[str]) -> None:
    os.makedirs(embedding_dir, exist_ok=True)

    if args.create_index:
        password = os.getenv("ELASTIC_SEARCH_UNI_PASSWORD")
        if not password:
            raise RuntimeError("ELASTIC_SEARCH_UNI_PASSWORD is not set")
        create_index(password, args.index, args.dims, args.shards)
        logging.info(f"Index ensured/created: {args.index}")

    total_archives = len(unprocessed_archives)

    process_parent_dir(parent_dir)

    for idx, file_path in enumerate(unprocessed_archives, start=1):
        unpack_tar_gz(file_path, parent_dir)

        process_parent_dir(parent_dir)

        progress = (idx / total_archives) * 100 if total_archives else 100
        logging.info(f"Progress: {progress:.2f}%")

if __name__ == '__main__':
    main()
