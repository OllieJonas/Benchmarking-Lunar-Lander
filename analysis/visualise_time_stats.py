import os

from datetime import datetime
from argparse import ArgumentParser

import pyprof2calltree
from cProfile import Profile

DATE_TIME_FORMAT = "%H-%M-%S_%m-%d-%Y"


def main(args):
    file = args.input

    if args.use_latest:
        file = f"{trailing_slash(args.directory)}{get_latest_run_of(args.directory, args.name)}logs/{args.name}"

    if file is None:
        raise ValueError("File not found! If you're using input path, please check it's correct. If you're using "
                         "latest, then this program is unable to find a latest run! :(")


def get_latest_run_of(root: str, name: str):
    """
    copied from rlcw code - would like to be able to use this as a standalone project &
    cba to deal with importing business from rlcw
    """
    walk = list(os.walk(root))[1:]
    potential_candidates = sorted({s[0].split("/").pop().split("\\")[0] for s in walk if name in s[0]},
                                  key=lambda s: datetime.strptime(s.split(" - ")[1].strip(), DATE_TIME_FORMAT),
                                  reverse=True)
    return trailing_slash(potential_candidates[0]) if potential_candidates else ""


def trailing_slash(s: str):
    return s if s.endswith("/") else f"{s}/"


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("-f", "--file", type=str, help="Input file (excl. path)", default="")
    argument_parser.add_argument("-d", "-directory", type=str, help="Root directory to find runs", default="out/")
    argument_parser.add_argument("-n", "--name", type=str, help="Name of the agent", default="ddpg")
    argument_parser.add_argument("-e", "--extension", type=str, help="File extension for dump file", default=".dmp")
    argument_parser.add_argument("-s", "--sort", type=str, help="What to sort the time stuff by", default="cumtime")
    argument_parser.add_argument("-l", "--use-latest", action="store_true", help="Toggles whether to use the "
                                                                                 "latest run of an agent", default=True)
    argument_parser.add_argument("-s", "--save", action="store_true", help="Whether to store a "
                                                                           ".kgrind file for later", default=True)
    argument_parser.add_argument("-v", "--visualise", action="store_true", help="Whether to render a UI", default=True)

    args = argument_parser.parse_args()
    main(args)
