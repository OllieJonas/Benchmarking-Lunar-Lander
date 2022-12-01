import abc
import numpy as np

from IPython import display
from matplotlib import pyplot as plt


import logger
import util
from orchestrator import Results
from replay_buffer import ReplayBuffer
from agents.abstract_agent import CheckpointAgent


class RunnerFactory:

    def __init__(self):
        self.LOGGER = logger.init_logger("RunnerFactory")

    def get_runner(self, env, agent, seed: int, should_render, episodes_to_save, max_timesteps,
                   max_episodes, start_training_timesteps, training_ctx_capacity,
                   should_save_checkpoints, checkpoint_history):
        if isinstance(agent, CheckpointAgent):
            return CheckpointRunner(env=env, agent=agent, seed=seed, should_render=should_render,
                                    episodes_to_save=episodes_to_save, max_timesteps=max_timesteps,
                                    max_episodes=max_episodes, start_training_timesteps=start_training_timesteps,
                                    training_ctx_capacity=training_ctx_capacity,
                                    should_checkpoint=should_save_checkpoints,
                                    checkpoint_history=checkpoint_history)
        else:
            if should_save_checkpoints:
                self.LOGGER.warning("You can't save checkpoints for this agent!")

            return SimpleRunner(env=env, agent=agent, seed=seed, should_render=should_render,
                                episodes_to_save=episodes_to_save, max_timesteps=max_timesteps,
                                max_episodes=max_episodes, start_training_timesteps=start_training_timesteps,
                                training_ctx_capacity=training_ctx_capacity)


class RunnerStrategy(abc.ABC):

    def __init__(self, env, agent, seed: int, should_render, episodes_to_save, max_timesteps,
                 max_episodes, start_training_timesteps, training_ctx_capacity):
        self.LOGGER = logger.init_logger("Runner")

        self.env = env
        self.agent = agent

        self.seed = seed

        self.should_render = should_render
        self.episodes_to_save = episodes_to_save

        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes

        self.start_training_timesteps = start_training_timesteps
        self.training_ctx_capacity = training_ctx_capacity

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError("Not implemented!")

    def render(self):
        if self.should_render:
            is_using_jupyter = util.is_using_jupyter()

            if is_using_jupyter:
                image = plt.imshow(self.env.render())
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                self.env.render()


# If your agent inherits CheckpointAgent, it's using this runner.
class CheckpointRunner(RunnerStrategy):

    def __init__(self, env, agent, seed: int, should_render, episodes_to_save, max_timesteps,
                 max_episodes, start_training_timesteps, training_ctx_capacity, should_checkpoint, checkpoint_history):
        super().__init__(env, agent, seed, should_render, episodes_to_save, max_timesteps, max_episodes,
                         start_training_timesteps, training_ctx_capacity)
        self.should_checkpoint = should_checkpoint
        self.checkpoint_history = checkpoint_history

    def run(self):
        training_context = ReplayBuffer(self.training_ctx_capacity)
        results = Results(agent_name=self.agent.name(), date_time=util.CURR_DATE_TIME)

        timesteps_count = 0
        curr_best_score = util.MIN_INT

        cumulative_rewards = []

        for episode_no in range(self.max_episodes):
            if timesteps_count >= self.max_timesteps:
                break

            state, info = self.env.reset()

            terminated = False
            truncated = False
            cumulative_reward = 0

            while not terminated or truncated:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.render()

                training_context.add(state, next_state, action, reward, int(terminated))

                timestep_result = Results.Timestep(state=state, action=action, reward=reward)
                summary = results.add(episode_no, timestep_result, episode_no in self.episodes_to_save)

                if summary is not None:
                    self.LOGGER.info(
                        f"Episode Summary for {episode_no} (Cumulative, Avg, No Timesteps): {summary}")

                cumulative_reward += reward
                state = next_state
                timesteps_count += 1

            cumulative_rewards.append(cumulative_reward)
            average_score = np.average(cumulative_rewards[-self.checkpoint_history:])

            if self.should_checkpoint and average_score > curr_best_score:
                curr_best_score = average_score
                self.agent.save()

        return results


class SimpleRunner(RunnerStrategy):

    def __init__(self, env, agent, seed: int, should_render, episodes_to_save, max_timesteps,
                 max_episodes, start_training_timesteps, training_ctx_capacity):
        super().__init__(env, agent, seed, should_render, episodes_to_save, max_timesteps,
                         max_episodes, start_training_timesteps, training_ctx_capacity)

    def run(self):
        training_context = ReplayBuffer(self.training_ctx_capacity)
        results = Results(agent_name=self.agent.name(), date_time=util.CURR_DATE_TIME)

        curr_episode = 0
        state, info = self.env.reset()

        for t in range(self.max_timesteps):
            if curr_episode > self.max_episodes:
                break

            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.render()

            training_context.add(state, next_state, action, reward, int(terminated))

            if t > self.start_training_timesteps:
                self.agent.train(training_context)

            state = next_state

            timestep_result = Results.Timestep(state=state, action=action, reward=reward)
            summary = results.add(curr_episode, timestep_result, curr_episode in self.episodes_to_save)

            if summary is not None:
                self.LOGGER.info(f"Episode Summary for {curr_episode - 1} (Cumulative, Avg, No Timesteps): {summary}")

            # self.LOGGER.debug(timestep_result)

            if terminated:
                curr_episode += 1
                state, info = self.env.reset()

            if truncated:
                state, info = self.env.reset()

        return results
