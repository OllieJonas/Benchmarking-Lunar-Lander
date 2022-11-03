# Creating an Agent

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