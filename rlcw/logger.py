import logging
import util
import sys

logger_level = logging.INFO


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
    logs_dir = f'{util.get_curr_session_output_path()}logs'
    util.make_dir(logs_dir)

    logging.basicConfig(filename=f'{logs_dir}/stdout.log', level=logger_level)

    logger = logging.getLogger(f'{NAME} {suffix}')

    logger.setLevel(logger_level)

    handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
