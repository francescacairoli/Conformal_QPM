# Author: Tom Kuipers, King's College London
# Modified version of PettingZoo MPE Simple Adversary environment.
# Main modifications: alteration of adversary observation space to match agent
#                     addition of sensor noise etc.
# New observation is padded with zeros.
# TODO: Cleanup and restructure.

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

from typing import Optional
from cqpm_mpe.config import *


class raw_env(SimpleEnv, EzPickle):
    # Need to set max_cycles to 1 otherwise tianshou will not run properly
    # EDIT: num of agents, adveraries and landmarks passed into environment
    def __init__(
        self,
        seed_w: int,
        seed_n: int,
        max_cycles: Optional[int] = 25,
        continuous_actions: Optional[bool] = False,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        EzPickle.__init__(self, CONFIG[CFG_ENV]['n_agents'], CONFIG[CFG_ENV]['n_adversaries'], CONFIG[CFG_ENV]['n_landmarks'], max_cycles, continuous_actions, CONFIG[CFG_SIM]['render_mode'])
        scenario = Scenario(seed_w, seed_n, initial_state=initial_state)
        world = scenario.make_world(CONFIG[CFG_ENV]['n_agents'], CONFIG[CFG_ENV]['n_adversaries'], CONFIG[CFG_ENV]['n_landmarks'])
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=CONFIG[CFG_SIM]['render_mode'],
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_adversary_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):

    def __init__(self, world_seed, noise_seed, initial_state=None):
        super().__init__()
        self.world_seed = world_seed
        self.noise_seed = noise_seed
        self.initial_state = initial_state
        self.world_rng = np.random.default_rng(seed=self.world_seed)
        self.has_reset = True
        self.agent_initial = None

    def make_world(self, n_agents=2, n_adversaries=1, n_landmarks=2):
        world = World()
        # set any world properties first
        # EDIT: Inplemented customisable number of agents, adversaries and landmarks
        world.dim_c = 2
        num_agents = n_agents + n_adversaries
        world.num_agents = num_agents
        num_adversaries = n_adversaries
        num_landmarks = n_landmarks
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            # NOTE: Edited world's agent list to put agents first
            agent.adversary = False if i < n_agents else True
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < n_agents else i - n_agents
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        return world
    
    def random_reset(self, world):
        # Set goal landmark to seed modulo number of landmarks
        goal = world.landmarks[self.world_seed % CONFIG[CFG_ENV]['n_landmarks']]
        goal.color = np.array([0.15, 0.65, 0.15])
        # Set the target goal of every agent to the new goal
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.world_rng.uniform(-3, +3, world.dim_p)
            #agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel = self.world_rng.uniform(-0.5, +0.5, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.world_rng.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def load_initial_state(self, world):
        #[ pos, vel, pos, vel, pos, vel, landmark1, landmark2]
        goal_idx = self.world_seed % CONFIG[CFG_ENV]['n_landmarks']
        agent_bound = int(len(world.agents) * 4)
        # Agent pos
        for i in range(0, agent_bound, 4):
            agent = world.agents[i // 4]
            agent.state.p_pos = self.initial_state[i:(i + 2)]
            agent.state.p_vel = self.initial_state[(i + 2):(i + 4)]
            agent.state.c = np.zeros(world.dim_c)
        # Landmark pos
        for i in range(agent_bound, len(self.initial_state), 2):
            idx = (i - (agent_bound + 2)) // 2
            landmark = world.landmarks[idx]
            landmark_pos = self.initial_state[i:(i + 2)]
            landmark.state.p_pos = landmark_pos
            # Make sure velocity of landmark is 0.
            # Depending on case study, this may change later.
            landmark.state.p_vel = np.zeros(world.dim_p)
        for agent in world.agents:
            agent.goal_a = world.landmarks[goal_idx]


    def reset_world(self, world, np_random):
        self.has_reset = True
        self.agent_initial = [ True ] * world.num_agents
        if type(self.initial_state) is np.ndarray:
            self.load_initial_state(world)
        else:
            self.random_reset(world)

        # Set the colours etc of the agents when resetting env
        for agent in world.agents:
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])

        # Set colour of the goal
        world.agents[0].goal_a.color = np.array([0.15, 0.65, 0.15])

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for lm in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
            dists.append(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    # def agent_reward(self, agent, world):
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it

    #     # Get all other good agents and adversary agents
    #     good_agents = self.good_agents(world)
    #     adversary_agents = self.adversaries(world)

    #     # Calculate positive reward for agents
    #     dist_goal = np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos)
    #     avg_dist_goal = np.average([ np.linalg.norm(a.state.p_pos - agent.goal_a.state.p_pos) for a in good_agents ])
    #     pos_rew = -dist_goal + (avg_dist_goal / 2)

    #     # Calculate negative reward for adversary
    #     min_dist_adv = min(
    #         np.linalg.norm(a.state.p_pos - agent.goal_a.state.p_pos)
    #         for a in adversary_agents
    #     )
    #     adv_rew = min_dist_adv
        
    #     return (pos_rew + adv_rew) / len(good_agents)

    # def adversary_reward(self, agent, world):
    #     # Rewarded based on proximity to the goal landmark
        
    #     good_agents = self.good_agents(world)
    #     adversary_agents = self.adversaries(world)

    #     dist_goal = np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos)
    #     avg_dist_goal = np.average(np.array([ np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos) for a in good_agents ]))
    #     dist_agent = sum(np.linalg.norm(agent.state.p_pos - a.state.p_pos) for a in good_agents)

    #     return -dist_goal + avg_dist_goal

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in adversary_agents
            )
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if (
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    < 2 * a.goal_a.size
                ):
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if (
                min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    for a in good_agents
                )
                < 2 * agent.goal_a.size
            ):
                pos_rew += 5
            pos_rew -= min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sqrt(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if (
                np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
                < 2 * agent.goal_a.size
            ):
                adv_rew += 5
            return adv_rew

    def observation(self, agent, world):

        agent_idx = world.agents.index(agent)

        # get positions of all entities in this agent's reference frame
        entity_pos = np.array( [ entity.state.p_pos - agent.state.p_pos for entity in world.landmarks ] ).flatten()
        
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        other_pos = np.array(other_pos).flatten()
        
        # Just agent pos/vel, other pos and landmark pos
        obs = np.concatenate(
                (agent.state.p_pos, agent.state.p_vel, other_pos, entity_pos)
            ).flatten()

        self.agent_initial[agent_idx] = False
        
        return obs
