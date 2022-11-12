# Saving Files

## Project Structure

The Project Structure is as follows (with example output, "out" and everything underneath 
will be creating upon running the program):

    - root
        - docs
        - out
            - <agent-name> - <time of run>
                - logs
                - policies
                - results
                    - csv
                    - png
                    - raw
                    - recordings
        - rlcw
            - agents
            - notebooks
            - tests
        - scripts

I'll go through each one and what it does.

### Docs

Where any documentation for the project is kept (hey here we are!)

### Out

Out contains anything outputted by the program upon running. Upon startup, it will create "out" if it doesn't exist, as 
well as a new directory containing any output from that particular running of the program.

If you turn everything on, it will create the following:

        - logs
            - stdout.log
        - policies
        - results
            - csv
                - results.csv
            - png
                - average_rewards.png
                - cumulative_rewards.png
                - no_timesteps.png
            - raw
                - results
            - recordings
                - rl-video-episode-X.mp4
                - rl-video-episode-X.meta.json

The results.csv file contains the average and cumulative rewards, and number of timesteps for each episode.

The png directory contains charts for these three things.

I think the rest is pretty self-explanatory.

_Please note, if you're using Docker, the run.bat script will copy it out of the container by default, 
so this should still appear._

#### rlcw

This is our source code. Please write code in here :) I'll write a brief description of what each thing contains.

##### agents

Contains the abstract template of an agent, as well as any implementations. Please put your agents in here!

##### aotebooks

Simply for Docker and Jupyter as an experiment for rendering using Docker - don't worry about it too much.

##### tests

Any unit tests we want to write should go in here.

### scripts

This just contains the entrypoint for the Docker image - please don't touch anything in there unless you know what
you're doing!