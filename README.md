# Benchmarking Different RL Methods on Lunar Lander
[![PythonVersion][python-version]][python-home]
[![GymVersion][gym-version]][gym-version]
[![License][license]](LICENSE)


[gym-version]: https://img.shields.io/badge/gym-0.26.2-blue
[python-version]: https://img.shields.io/badge/python-3.5, 3.6, 3.7, 3.8, 3.9-blue
[license]: https://img.shields.io/badge/license-MIT-green

[python-home]: https://python.org
[gym-website]: https://github.com/openai/gym

<p align="center">
  <img src="https://imgur.com/ODEnfxR.gif" width="50%" />
</p>

Submission for CM30225 (Reinforcement Learning) at the University of Bath, written by [Fraser Dwyer](https://github.com/Fraser-Dwyer), [Helen Harmer](https://github.com/helen2000), [Ollie Jonas](https://github.com/OllieJonas), and [Yatin Tanna](https://github.com/YatinTanna).

This project aims to provide a framework for multiple different RL methods, and provide some utilities that are common amongst all of them.

A more detailed description of the project (including the config file, creating an agent and the project's structure) can be found in the `docs/` directory.

## Features

Brief outline of the features provided:

- Automatically provides runner code with a replay buffer (with conversion to PyTorch tensors)
- Output (raw data & charts and logs) of overall summary of project (cumulative reward, average reward, no timesteps) per episode
- Output (raw data & charts) of individual rewards for each timestep at specified episodes
- Output recordings of specified episodes
- Saving of checkpoints for neural networks at specified intervals
- Loading of neural network parameters at startup (either from absolute path, relative path or from latest run)
- Swaps between continuous and discrete action spaces of LunarLander at runtime
- Provides easy-to-read configuration file, which dynamically loads a section for each agent, to allow specifying of different hyper-parameters

## Limitations

A brief outline of either things this program can't do / things you really have to fight the program to achieve (that we wish it could do)

- Log stdout / stderr to an output file (it only logs what we log, **not** what gym logs)
- Multiple runner implementations (we use a different one for SARSA, but it's very ugly code)
- Save episodes based on some criteria that's found in run-time (for example: DQN had some runs which took tens of thousands of time-steps to complete, but we have no way of specifying to save recordings of those episodes - you **have** to specify which episodes to save at compile-time)

## Installation Guide

### Console (Linux / Mac)
  
For Linux / Mac, it's very easy to do:

1. Navigate to the root directory for this project
2. Run pip3 install -r requirements.txt
3. Run pip3 install swig
4. Run pip3 install gym[All] or pip3 install gym[Box2D]
5. Set your PYTHONPATH environment variable to rlcw

### Windows

For Windows, you can run this program using Docker.
  
#### Installation Guide (Windows)
  
  1. Install Docker. You can find the link for this here: [Install Docker](https://docs.docker.com/get-docker/ "Docker")
  2. For Windows, you're going to need to use WSL 2 Linux Kernel (A Linux Kernel for Windows), and install the Ubuntu distro for WSL. This guide might be helpful:  [Install WS2](https://learn.microsoft.com/en-us/windows/wsl/install-manual). Also note that Docker Desktop _will automatically start when you start your PC._ If you want to disable this, do the following:
      1. Open Task Manager
      2. Go to the Startup Tab
      3. Find Docker Desktop, right click and click Disable.

## Running the Program

For UNIX-based systems, you just need to run the program like any old python program: `python3 -m main`. 
  
For Windows, a `run.bat` file has been included for convenience sake in the root directory. This builds and runs the image, and then collects any results from the container. 
  
