import logging
import sys
import os

from datetime import datetime

CURR_DATE_TIME = f'{str(datetime.now().strftime("%H-%M-%S_%m-%d-%Y"))}'

using_jupyter = False

logger_level = logging.INFO


def make_dir(name: str):
    if not os.path.exists(name):
        os.mkdir(name)


def set_using_jupyter(value: bool):
    global using_jupyter
    using_jupyter = value


def is_using_jupyter():
    return using_jupyter


def set_logger_level(level):
    global logger_level
    logger_level = level


def get_root_output_path():
    return '../../out/' if using_jupyter else '../out/'


def save_file(path):
    pass


def init_logger(suffix: str = "") -> logging.Logger:
    """
    Initialises a logger. Loggers by default start with the name "RL-CW", then have a suffix. I'm thinking maybe have
    the name of the algorithm we're implementing here?

    Please only call this once in a given file, then store it for everything else.
    """
    NAME = "RL-CW"
    logs_dir = f'{get_root_output_path()}logs'
    make_dir(logs_dir)

    logging.basicConfig(filename=f'{logs_dir}/{CURR_DATE_TIME}.log', level=logger_level)

    logger = logging.getLogger(f'{NAME} {suffix}')

    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
