# Reinforcement Learning Coursework (2022)


## Installation Guide

So basically, installing Gym itself is fine, but for some reason Windows HATES one package in the environment, Box2D.

To download this repository (all OS):

1) Run git clone https://github.com/OllieJonas/ReinforcementLearningCW <your directory>

For Linux / Mac, it looks like its quite easy to do. I need to check this by installing it on my laptop (it runs Linux, I'll update this space as I do it), you just need to do the following:

### Console (Mac / Linux)
1) Navigate to the root directory for this project.
2) Run pip3 install -r requirements.txt
3) Run pip3 install gym[All] or pip3 install gym[Box2D]

### Windows (or All if this doesn't work)

For Windows, I've come up with a MASSIVE workaround for this, which is to use something called Docker.
