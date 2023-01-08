import logger
import util
from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer
from results import Results


# add memory profiler to end of each episode
class Runner(object):

    def __init__(self, env, agent, seed: int, should_render, episodes_to_save, max_timesteps, max_ep_timestep,
                 max_episodes, start_training_timesteps, training_ctx_capacity, should_save_checkpoints,
                 save_every, should_invert_done, verbose=False):
        self.LOGGER = logger.init_logger("Runner")

        self.env = env
        self.agent = agent

        self.seed = seed

        self.should_render = should_render
        self.episodes_to_save = episodes_to_save

        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.max_ep_timestep = max_ep_timestep

        self.start_training_timesteps = start_training_timesteps
        self.training_ctx_capacity = training_ctx_capacity

        self.should_save_checkpoints = should_save_checkpoints
        self.save_every = save_every

        self.should_invert_done = should_invert_done

        self.verbose = verbose

        self.is_eligible_for_checkpoints = isinstance(agent, CheckpointAgent)

        if agent.name() == "sarsa":
            self.run_agent = self.run_sarsa
        else:
            self.run_agent = self.run
    
    def run_sarsa(self):
        
        training_context = ReplayBuffer(self.training_ctx_capacity,
                                        is_continuous=self.agent.requires_continuous_action_space)
        results = Results(agent_name=self.agent.name(), date_time=util.CURR_DATE_TIME)

        state, info = self.env.reset()
        action = self.agent.get_action(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)

        curr_episode = 0
        ep_timestep = 0
        for t in range(self.max_timesteps):
            if curr_episode > self.max_episodes:
                break
            
            next_action = self.agent.get_action(next_state)
            nn_state, reward_1, terminal_1, _, _ = self.env.step(next_action)

            # render
            if self.should_render:
                self.env.render()

            training_context.add_to_sarsa(
                state,
                next_state,
                action,
                next_action,
                reward,
                int(terminated))

            if t > self.start_training_timesteps:
                self.agent.train(training_context)

            timestep_result = Results.Timestep(state=state, action=action, reward=reward)
            _summary = results.add(curr_episode, timestep_result, curr_episode in self.episodes_to_save)

            if _summary is not None:
                self.LOGGER.info(f"Episode Summary for {curr_episode - 1} (Cumulative, Avg, No Timesteps): {_summary} epsilon {self.agent.epsilon}")

            # self.LOGGER.debug(timestep_result)
            terminated = terminal_1

            if ep_timestep >1000:
                terminated = True

            if terminated:
                curr_episode += 1
                next_action = self.agent.get_action(next_state)
                training_context.add_to_sarsa(
                    state,
                    next_state,
                    action,
                    next_action,
                    reward,
                    int(terminated))

                state, info = self.env.reset()
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # decays epsilon 
                self.agent.decay_epsilon()
                ep_timestep = 0
            else:
                state = next_state
                next_state = nn_state
                action = next_action
                reward = reward_1
                ep_timestep +=1

            if truncated:
                state, info = self.env.reset()

            if self.is_eligible_for_checkpoints and self.should_save_checkpoints and \
                    curr_episode % self.save_every == 0:
                self.agent.save()

        return results


    def run(self):
        state, info = self.env.reset()
        training_context = ReplayBuffer(self.training_ctx_capacity,
                                        is_continuous=self.agent.requires_continuous_action_space)
        results = Results(agent_name=self.agent.name(),
                          date_time=util.CURR_DATE_TIME)

        curr_episode = 0
        ep_t = 0

        for t in range(self.max_timesteps):
            ep_t += 1
            if curr_episode > self.max_episodes:
                break

            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(
                action)

            # render
            if self.should_render:
                self.env.render()

            training_context.add(
                state,
                next_state,
                action,
                reward,
                int(terminated), invert_done=self.should_invert_done)

            if t > self.start_training_timesteps:
                self.agent.train(training_context)

            state = next_state

            timestep_result = Results.Timestep(
                state=state, action=action, reward=reward)
            _summary = results.add(
                curr_episode, timestep_result, curr_episode in self.episodes_to_save)

            if _summary is not None:
                self.LOGGER.info(
                    f"Episode Summary for {curr_episode - 1} (Cumulative, Avg, No Timesteps): {_summary}")

            # self.LOGGER.debug(timestep_result)

            if terminated or (ep_t > self.max_ep_timestep and self.max_ep_timestep != -1):
                ep_t = 0
                curr_episode += 1
                state, info = self.env.reset()

            if truncated:
                state, info = self.env.reset()

            if self.is_eligible_for_checkpoints and self.should_save_checkpoints and \
                    curr_episode % self.save_every == 0:
                self.agent.save()

        return results