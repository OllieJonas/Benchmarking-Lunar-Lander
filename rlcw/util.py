import logging
import os
import sys
import copy
import torch

from datetime import datetime

CURR_DATE_TIME = None
AGENT_NAME = None

using_jupyter = False

logger_level = logging.INFO


def with_file_extension(name, extension):
    extension = extension if extension.startswith(".") else f".{extension}"
    return f'name {"" if name.endswith(extension) else extension}'


def make_dir(name: str):
    if not os.path.exists(name):
        os.mkdir(name)


# -------------------------------- PATHS --------------------------------
def get_project_root_path():
    return f'{"/".join(copy.copy(sys.argv[0].split("/"))[:-2])}/'


def get_output_root_path():
    return f'{get_project_root_path()}out/'


def get_curr_session_output_path():
    global CURR_DATE_TIME

    if CURR_DATE_TIME is None:
        CURR_DATE_TIME = f'{str(datetime.now().strftime("%H-%M-%S_%m-%d-%Y"))}'

    return f'{get_output_root_path()}{AGENT_NAME} - {CURR_DATE_TIME}/'


# -------------------------------- SAVING AND LOADING --------------------------------

def save_file(directory, file_name, contents):
    with open(f'{get_curr_session_output_path()}/{directory}/{file_name}', 'w') as f:
        f.write(contents)


def save_torch_nn(net, file_name):
    torch.save(net.state_dict(), with_file_extension(file_name, ".mdl"))


def load_torch_nn(net, file_name):
    net.load_state_dict(torch.load(file_name))


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_using_jupyter():
    return using_jupyter


def set_using_jupyter(value: bool):
    global using_jupyter
    using_jupyter = value


def set_agent_name(agent_name):
    global AGENT_NAME
    AGENT_NAME = agent_name


def set_logger_level(level):
    global logger_level
    logger_level = level


def init_logger(suffix: str = "") -> logging.Logger:
    """
    Initialises a logger. Loggers by default start with the name "RL-CW", then have a suffix. I'm thinking maybe have
    the name of the algorithm we're implementing here?

    Please only call this once in a given file, then store it for everything else.
    """
    NAME = "RL-CW"
    logs_dir = f'{get_curr_session_output_path()}logs'
    make_dir(logs_dir)

    logging.basicConfig(filename=f'{logs_dir}/stdout.log', level=logger_level)

    logger = logging.getLogger(f'{NAME} {suffix}')

    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
