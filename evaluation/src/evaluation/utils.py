from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
from typing import List, Tuple
from src.constants import OPENAI_MODEL_LIST
import subprocess
import socket
import signal
import time


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def run_sglang_server(model_path, port, chat_template, result_save_path, tokenizer_path=None):
    sglang_command = f"python -m sglang.launch_server --model-path {model_path}" + \
                      f" --port {port} --enable-flashinfer --attention-reduce-in-fp32"
    
    if tokenizer_path is not None:
        sglang_command += f" --tokenizer-path {tokenizer_path}"

    if chat_template is not None:
        sglang_command += f" --chat-template {chat_template}"
    

    with open(result_save_path, 'w', encoding='utf-8') as log_file:
        pass
    log_file = open(result_save_path, 'a', encoding='utf-8')

    ## somehow this will hang if we don't output things to a FILE
    sglang_process = subprocess.Popen(
        sglang_command,
        shell=True,
        start_new_session=True,
        stdin=log_file,
        stdout=log_file,
        stderr=log_file,
        text=True
    )

    # Kill the subprocess when the parent process is killed
    def kill_process():
        sglang_process.kill()

    signal.signal(signal.SIGINT, kill_process)
    signal.signal(signal.SIGTERM, kill_process)

    # Read the subprocess output line by line
    seen_line_idx = 0
    while True:
        sglang_status = False
        # read from log_file instead of stdout
        with open(result_save_path, 'r', encoding='utf-8') as fread:
            lines = fread.readlines()
            lines = lines[seen_line_idx:]
            for line in lines:
                print(line, end='')
                seen_line_idx += 1
                if "200 OK" in line:
                    sglang_status = True
                    break
                else:
                    time.sleep(0.2)

        # if sglang server is ready, break the loop
        if sglang_status:
            break
    
    print("sglang server is ready.")
    return sglang_process


def _has_model_weights(dir_path: str):
    for files in Path(dir_path).iterdir():
        if files.is_file() and files.suffix == ".safetensors":
            return True
    return False


def find_checkpoint_paths(base_path: str) -> List[Tuple[Path, str]]:
    found_dirs = []
    # first check if its openai mode
    if base_path in OPENAI_MODEL_LIST or '/' not in base_path:  # probably remote model
        found_dirs.append((base_path, ""))
        return found_dirs
    
    # then check if it is a HF model
    hf_api = HfApi()
    try:
        _ = hf_api.model_info(base_path)
        found_dirs.append((base_path, ""))
    except (RepositoryNotFoundError, HFValidationError):
        # it is a local model
        if _has_model_weights(base_path):
            found_dirs.append((base_path, ""))

        for folder in Path(base_path).iterdir():
            if folder.is_dir():
                if _has_model_weights(folder):
                    found_dirs.append((folder, f"_{folder.name}"))
    return found_dirs