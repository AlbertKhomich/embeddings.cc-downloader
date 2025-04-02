import re
import sys
import os
import logging
import subprocess

def only_unextracted(embedding_dir, log_file=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            processed = set()
            if log_file and os.path.exists(log_file):
                extraction_pattern = re.compile(r"Successfully extracted\s+(.*?)\s+to\s+(.*)")

                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        m = extraction_pattern.search(line)
                        if m:
                            archive_path = m.group(1)
                            archive_name = os.path.basename(archive_path)
                            processed.add(archive_name)

            all_archives = [f for f in os.listdir(embedding_dir) if f.endswith('.tar.gz')]

            if processed:
                unprocessed_archives = [os.path.join(embedding_dir, a) for a in all_archives if a not in processed]
            else:
                unprocessed_archives = [os.path.join(embedding_dir,a ) for a in all_archives]

            logging.info("Unprocessed archives: %s", unprocessed_archives)
            return func(unprocessed_archives, *args, **kwargs)
        return wrapper
    return decorator

def parse_log(filename):
    extraction_pattern = re.compile(r"Successfully extracted\s+(.*?)\s+to\s+(.*)")
    file_path_pattern = re.compile(r"File paths:\s+(.*)")
    error_pattern = re.compile(r"Error adding")

    last_archive = None
    last_dest_dir = None
    current_folder = None
    commands = set()

    with open(filename, "r") as log_file:
        for line in log_file:
            line = line.strip()

            m_ext = extraction_pattern.search(line)
            if m_ext:
                last_archive = m_ext.group(1)
                last_dest_dir = m_ext.group(2)

            m_fp = file_path_pattern.search(line)
            if m_fp:
                paths_str = m_fp.group(1)
                m_folder = re.search(r"'entity_to_idx\.p':\s*'([^']+)", paths_str)
                if m_folder:
                    full_path = m_folder.group(1)
                    if full_path.endswith("/entity_to_idx.p"):
                        current_folder = full_path[:-len("/entity_to_idx.p")]
                        folder_name = current_folder.split("/")[-1]
                    else:
                        current_folder = None
                        folder_name = None

            if error_pattern.search(line):
                if current_folder and last_archive and last_dest_dir:
                    folder_name = current_folder.split("/")[-1]
                    cmd = f"tar -xvzf {last_archive} -C {last_dest_dir} {folder_name}"
                    commands.add(cmd)
                    current_folder = None

    return commands

def execute_commands(commands):
    for cmd  in commands:
        print(f"Executing command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Success: {result.stdout}")
        else:
            print(f"Error: {result.stderr}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    cmds = parse_log(log_file)

    if cmds:
        execute_commands(cmds)
    else:
        print("No extraction commands found.")

