import logging
import sys
import os

from datetime import datetime

STANDARD_FILE_NAME = f'{str(datetime.now().strftime("%H-%M-%S_%m-%d-%Y"))}'


def make_dir(name: str, logger: logging.Logger = None):
    if not os.path.exists(name):
        if logger is not None:
            logger.info(f'Directory {name} doesn\'t exist! Creating it now...')
        os.mkdir(name)


def init_logger(suffix: str = "", level=logging.INFO) -> logging.Logger:
    """
    Initialises a logger. Loggers by default start with the name "RL-CW", then have a suffix. I'm thinking maybe have
    the name of the algorithm we're implementing here?

    Please only call this once in a given file, then store it for everything else.
    """
    NAME = "RL-CW"
    logs_dir = '../logs'
    make_dir(logs_dir)

    logging.basicConfig(filename=f'{logs_dir}/{STANDARD_FILE_NAME}.log', level=level)

    logger = logging.getLogger(f'{NAME} {suffix}')

    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
