# Creating an Agent

## How

Creating an agent that's recognised by the program is very easy...

1) Create a new .py file under the agents package. Call it whatever you like.
2) Have it implement the AbstractAgent class in abstract_agent.
3) Inherit the methods contained, and call the super constructor, passing action_space into your agent _(kinda messy, maybe should try to change that idk lmao)_

So far, it should look like this:

```
from rlcw.agents.abstract_agent import AbstractAgent

class YourAgent(AbstractAgent):

    def __init__(self, action_space):
        super.__init__(action_space)
        
    def get_action(self, observation):
        pass
        
    def train(self, training_context):
        pass
```

Here, you can implement those get_action and train methods, based on the information said later on.

4) Go to main.py
5) Import your agent
6) Find the method called `get_agent(name, action_space)`
7) Add the following to it:

    `elif name.lower() == "<your_agent>": return YourAgent(action_space)`

Bingo bango bongo we're done! The get_agent name uses the name found in config.yml, so if you want to run your new 
agent, change the name to whatever you put in (7) to it. 

## Abstract Agent

Abstract Agent is an abstract class to give a template for each new agent that we create - so that our Runner can 
recognise each new agent we create.

### Methods and Constructor Defined

There's only **two** methods defined that will actually NEED to be implemented:

1) `get_action(observation)`
2) `train(training_context)`

#### get_action(observation)

`get_action` is being called at the beginning of the timestep, and will determine what action it takes (well duh.) 

The observation is the thing defined in gym. If you read gym's documentation on it, you'll get more information.

The action space is already defined in the constructor, so you're able to call `self.action_space` to get stuff there.

#### train(training_context)

`train` is being called after the current timestep is greater than the start_training timesteps defined in the 
config.yml.

`training_context` is defined as a list of dict objects, where each dict contains information about each timestep, 
where the index of the list represents that timestep. 

This dict object looks like this:

`
{"curr_obsv": observation,
"next_obsv": next_observation,
"reward": reward,
"action": action
}
`

TODO: It might be more accurate to define next_obvs and curr and curr as prev? Might make more sense? Not sure
TODO (Definitely): Make the list bounded, and then delete old items. 


_If we need anything else, we can add it later, tbh I'm not really sure what we need right now so I did safe bets?_