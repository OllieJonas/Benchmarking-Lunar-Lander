# Benchmarking Different RL Methods on Lunar Lander (2022)

Submission for CM30225 (Reinforcement Learning) at the University of Bath, written by [Fraser Dwyer](https://github.com/Fraser-Dwyer), [Helen Harmer](https://github.com/helen2000), [Ollie Jonas](https://github.com/OllieJonas), and [Yatin Tanna](https://github.com/YatinTanna).

A description of the project (including the config file, creating an agent and the project's structure) can be found in the docs/ directory.

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

For UNIX-based systems, you just need to run the program like any old python program: python3 -m main. 
  
For Windows, a run.bat file has been included for convenience sake in the root directory. This builds and runs the image, and then collects any results from the container.
Occasionally, you may need to prune any hanging images. To do this, use the following: docker prune images --force.

If you ever need to add a dependency to the project, just add it to the requirements.txt - Docker will automatically sort it out for you.
  
