import re
import sys
import subprocess

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

