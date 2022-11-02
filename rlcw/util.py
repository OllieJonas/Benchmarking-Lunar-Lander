import logging
import sys
from datetime import datetime

STANDARD_FILE_NAME = f'{str(datetime.now().strftime("%H-%M-%S_%m-%d-%Y"))}'


def init_logger(suffix: str = "", level=logging.INFO) -> logging.Logger:
    """
    Initialises a logger. Loggers by default start with the name "RL-CW", then have a suffix. I'm thinking maybe have
    the name of the algorithm we're implementing here?

    Please only call this once in a given file, then store it for everything else.
    """
    NAME = "RL-CW"

    logging.basicConfig(filename=f'../logs/{STANDARD_FILE_NAME}.log', level=level)

    logger = logging.getLogger(f'{NAME} {suffix}')

    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
