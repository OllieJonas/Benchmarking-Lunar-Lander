import copy
import os
import sys
from datetime import datetime

CURR_DATE_TIME = None
DATE_TIME_FORMAT = "%H-%M-%S_%m-%d-%Y"
AGENT_NAME = None

using_jupyter = False


# -------------------------------- FILES --------------------------------

def with_file_extension(name, extension):
    extension = extension if extension.startswith(".") else f".{extension}"
    return f'{name}{"" if name.endswith(extension) else extension}'


def make_dir(name: str):
    if not os.path.exists(name):
        os.mkdir(name)


def get_latest_run_of(name: str):
    # it's not my code if I don't fit in a ridiculous, confusing & overly complicated one-liner >:)
    # shame im really fighting against python to do this, man streams are so much easier >:(
    walk = list(os.walk(get_output_root_path()))[1:]
    potential_candidates = sorted({s[0].split("/").pop().split("\\")[0] for s in walk if name in s[0]},
                                  key=lambda s: datetime.strptime(s.split(" - ")[1].strip(), DATE_TIME_FORMAT),
                                  reverse=True)
    return f"{potential_candidates[0]}/" if potential_candidates else ""


# -------------------------------- PATHS --------------------------------
def get_project_root_path():
    return f'{"/".join(copy.copy(sys.argv[0].split("/"))[:-2])}/'


def get_output_root_path():
    return f'{get_project_root_path()}out/'


def get_curr_session_output_path():
    global CURR_DATE_TIME

    if CURR_DATE_TIME is None:
        CURR_DATE_TIME = f'{str(datetime.now().strftime(DATE_TIME_FORMAT))}'

    return f'{get_output_root_path()}{AGENT_NAME} - {CURR_DATE_TIME}/'


# -------------------------------- SAVING AND LOADING --------------------------------

def save_file(directory, file_name, contents):
    with open(f'{get_curr_session_output_path()}/{directory}/{file_name}', 'w') as f:
        f.write(contents)


def set_agent_name(agent_name):
    global AGENT_NAME
    AGENT_NAME = agent_name


# -------------------------------- MATHS --------------------------------
MIN_INT = -2147483648  # just a small af number
MAX_INT = 2147483647  # just a large af number
