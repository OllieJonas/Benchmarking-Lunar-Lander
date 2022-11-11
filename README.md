# Reinforcement Learning Coursework (2022)


## Installation Guide

So basically, installing Gym itself is fine, but for some reason Windows HATES one package in the environment, Box2D.

To download this repository (all OS):

1) Run git clone https://github.com/OllieJonas/ReinforcementLearningCW <your directory>

### Console (Mac / Linux)
  
For Linux / Mac, it's very easy to do:

1. Navigate to the root directory for this project.
2. Run pip3 install -r requirements.txt
3. Run pip3 install swig
4. Run pip3 install gym[All] or pip3 install gym[Box2D]
5. Don't forget to set your PYTHONPATH env var to rlcw

### Windows (or All if Mac / Linux doesn't work)

For Windows, I've come up with a MASSIVE workaround for this, which is to use something called Docker.

#### What is Docker?

So Docker is a containerization application - ie. it allows you to build, run and manage containers.

Well, you may be asking what are containers now - seems pretty obvious. They're basically your program and its dependencies packaged together into one easily deployable bundle.


So the idea of Docker is to install the program and its dependencies onto an "image", which can then be deployed as many times as you like as a "container".

It's mainly used in the context of large-scale application deployment, where you're unsure of what the machine you're installing your software will be running. The idea is, as long as each machine is running Docker, it means you don't need to care about having a specific version of a library, or having the correct environment variables on there, stuff like that.

The details of this are specified in a Dockerfile, which tells you what to include, what to install and how to enter the application.

#### So why the hell are we using it then?

Well, although this is VERY wrong in terms of how Docker actually works or what it does, from our usage, we're essentially using it as a virtual machine with absolutely minimal installed (except for what we specify) that's only able to run a single thing (ie. our program). This means that when you run the program through Docker, you will essentially be running it on Linux with only the bare bones essentials for our project to work.
  
#### Installation Guide (Windows)

Now, unfortunately, Docker can be very confusing to install.
  
  1. Install Docker. You can find the link for this here: [Install Docker](https://docs.docker.com/get-docker/ "Docker")
  2. For Windows, you're going to need to use WSL 2 Linux Kernel (A Linux Kernel for Windows), and install the Ubuntu distro for WSL. For me personally, I had to follow this guide to install it: [Install WS2](https://learn.microsoft.com/en-us/windows/wsl/install-manual). Also note that Docker Desktop _will automatically start when you start your PC._ If you want to disable this, do the following:
      1. Open Task Manager
      2. Go to the Startup Tab
      3. Find Docker Desktop, right click and click Disable.

## Running the Program

For UNIX-based systems, you just need to run the program like any old python program (using main.py as the main module). 
  
For Windows, you need to run the run.bat file in the root directory. Running the run.bat script will start a Jupyter server, which by default you'll be able to access by going to [localhost:8888](http://localhost:8888). Check the console for more details.
  
### Jupyter vs Python

You are able to run this program from both a Jupyter server, and just as straight Python. The reason for this is that we're essentially using Jupyter as a hack to render the environment for Windows. None of us actually like Jupyter, so it's purely there to render - there's no actual substance in the notebook.

### Final Notes  
I've tried my best to set this project up such that we don't ever need to worry about the Docker part of it. Hopefully, once its installed, we won't need to worry about it ! :)

If you ever need to add a dependency to the project, just add it to the requirements.txt - Docker will automatically sort it out for you.
  
