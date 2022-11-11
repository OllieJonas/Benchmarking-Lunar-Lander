# The Configuration File
In the root directory, there is a config.yml, which allows you to change certain things about it.

Below, there is a table of each thing in the config, and what it does:

| Name           | Description                                            | Data Type | Notes                                                                                                   |
|----------------|--------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------------|
| **Overall**    |                                                        |           | Section                                                                                                 |
|                |                                                        |           |                                                                                                         |
| Agent_Name     | Specifies which agent (ie. algorithm) will be used     | String    |                                                                                                         |
| **Output**     |                                                        |           |                                                                                                         |
|                |                                                        |           |                                                                                                         |
| Verbose        | Whether to log debug statements to console / logs      | Boolean   |                                                                                                         |
| Render         | Whether to render a UI for the agent's training        | Boolean   |                                                                                                         |
| Save_Timesteps | Whether to save all timesteps as a file after training | Boolean   |                                                                                                         |
|                |                                                        |           |                                                                                                         |
| **Episodes**   |                                                        |           |                                                                                                         |
| Max            | Maximum amount of episodes before stopping training.   | Integer   | Training will stop based on whichever <br/> one comes first out of max_episodes <br/> and max_timesteps |
|                |                                                        |           |                                                                                                         |
| **Timesteps**  |                                                        |           |                                                                                                         |
| Max            | The total number of timesteps to iterate through       | Integer   |                                                                                                         |
| Start_Training | When to start allowing the agent to train              | Integer   | \> Timesteps (Max)                                                                                      |
|                |                                                        |           |                                                                                                         |