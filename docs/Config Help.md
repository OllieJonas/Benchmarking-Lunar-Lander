# The Configuration File
In the root directory, there is a config.yml, which allows you to change certain things about it.

Below, there is a table of each thing in the config, and what it does:

| Name                       | Description                                        | Data Type | Notes              |
|----------------------------|----------------------------------------------------|-----------|--------------------|
| Agent                      | Specifies which agent (ie. algorithm) will be used | String    |                    |
| Verbose                    | Whether to log debug statements to console / logs  | Boolean   |                    |
| Render                     | Whether to render a UI for the agent's training    | Boolean   |                    |
| Timesteps (max)            | The total number of timesteps to iterate through   | Integer   |                    |
| Timesteps (start_training) | When to start allowing the agent to train          | Integer   | \> Timesteps (Max) |
