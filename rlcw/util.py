import logging
import os
import sys
import copy
import torch
import numpy as np

from datetime import datetime

CURR_DATE_TIME = None
AGENT_NAME = None

using_jupyter = False


# -------------------------------- FILES --------------------------------

def with_file_extension(name, extension):
    extension = extension if extension.startswith(".") else f".{extension}"
    return f'name {"" if name.endswith(extension) else extension}'


def make_dir(name: str):
    if not os.path.exists(name):
        os.mkdir(name)


def get_latest_run_of(name: str):
    # it's not my code if I don't fit in a ridiculous, confusing & overly complicated one-liner >:)
    # shame im really fighting against python to do this, man streams are so much easier >:(
    walk = list(os.walk(get_output_root_path()))[1:]
    latest = sorted(list(set([s[0].split("/").pop().split("\\")[0] for s in walk if name in s[0]])), reverse=True)
    return [s for s in walk if latest and latest[0] in s[0]]


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


if __name__ == "__main__":
    print(get_latest_run_of("ddpg"))
    print(get_latest_run_of("sac"))
