from __future__ import annotations

from typing import List, Optional, Union, Tuple, Iterable, Any

import numpy as np
import torch

from blockchain_mdps import BlockchainModel, StateTransitions
from ..experience_acquisition.experience import Experience
from tianshou.utils.net.common import Net
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
import torch
# do not integrate this into the env, otherwise it will be slow
def load_dqn_model():
    state_shape = 46
    action_shape = 44
    hidden_sizes = [128, 128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
        # dueling=(Q_param, V_param),
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=5e-5)
    gamma = 0.99
    n_step = 2
    target_update_freq = 320
    policy = DQNPolicy(
        net,
        optim,
        gamma,
        n_step,
        target_update_freq=target_update_freq,
    )
    # ckpt_path = '/home/jys3649/projects/xai_masknet/selfish_mining/log/blockchainfee/dqn/ckpt1.pth'
    ckpt_path = '/home/zck7060/xrl4security/selfish_mining/baseline/model/checkpoint.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint["model"])
    # policy.load_state_dict(checkpoint)
    policy.set_eps(0.0)
    return policy
dqn_policy = load_dqn_model()

class MDPBlockchainSimulatorMaskNet:
    def __init__(self, blockchain_model: BlockchainModel, expected_horizon: int, check_valid_states: bool = False,
                 device: torch.device = torch.device('cpu'), include_transition_info: bool = False, policy=None):
        self._model = blockchain_model
        self.expected_horizon = expected_horizon

        self.check_valid_states = check_valid_states
        self.device = device
        self.include_transition_info = include_transition_info

        self.state_space = self._model.state_space
        self.num_of_states = self._model.state_space.size
        self.state_space_dim = self._model.state_space.dimension

        self.initial_state = self._model.initial_state
        self.final_state = self._model.final_state

        self.num_of_actions = self._model.action_space.size
        self.action_space = self._model.action_space

        self._current_state = self.initial_state
        self._prev_difficulty_contribution = 0
        self._cumulative_reward = 0
        self._cumulative_difficulty_contribution = 0
        self._episode_length = 0
        self._action_counts = {}
        self.observation_space = torch.ones(self.state_space_dim)
        self.action_space = torch.ones(self.num_of_actions)
        self.policy = policy
        self.lazy_reward = 0.15
        # self.lazy_reward = 0.05
        self.masks = 0
        self.total_length = 0
        self.mask_record_thres = 1000
        self.error_penalty = -1
        self.episode_mask = 0
        self.reset()

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self._model, self.expected_horizon, self.check_valid_states, self.device,
                                self.include_transition_info)

    def copy(self) -> MDPBlockchainSimulatorMaskNet:
        cls, params = self.__reduce__()
        return cls(*params)

    def enumerate_states(self) -> List[BlockchainModel.State]:
        return self._model.state_space.enumerate_elements()

    def enumerate_state_tensors(self) -> Iterable[torch.Tensor]:
        for state_tuple in self.enumerate_states():
            yield self.tuple_to_torch(state_tuple)

    def tuple_to_torch(self, state: BlockchainModel.State) -> torch.Tensor:
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def torch_to_tuple(self, state: torch.Tensor) -> BlockchainModel.State:
        state_tuple = tuple([int(value) for value in state.tolist()])
        return self._model.state_space.transform_element(state_tuple)

    def action_index_to_action(self, action_index: int) -> Any:
        return self._model.action_space.index_to_element(action_index)

    def get_state_transition_values(self, state: BlockchainModel.State, action: int) -> StateTransitions:
        return self._model.get_state_transitions(state, self.action_index_to_action(action),
                                                 check_valid=self.check_valid_states)

    def is_initial_state(self, state: Union[BlockchainModel.State, torch.Tensor]) -> bool:
        if torch.is_tensor(state):
            state = self.torch_to_tuple(state)
        return state == self.initial_state

    def step(self, action: int) -> Experience:
        self.total_length+=1
        mask_this_step = False
        if self._current_state == self.final_state:
            raise ValueError('Simulation ended')

        legal_actions = self.get_state_legal_actions_tensor(self._current_state)
        # print(len(np.where(legal_actions)))
        batch = Batch(obs=self.tuple_to_torch(self._current_state).reshape((1,-1)), info=None)
        agent_action = dqn_policy(batch).act[0]
        if action==0:
            real_action = agent_action
        else:
            real_action = np.random.choice(np.where(legal_actions)[0])
            mask_this_step = True
            self.masks += 1
            self.episode_mask += 1
            if self.total_length >= self.mask_record_thres:
                print("The mask ratio is: ", self.masks / self.total_length)
                self.total_length = 0
                self.masks = 0

        if not legal_actions[real_action]:
            real_action = np.random.choice(np.where(legal_actions)[0])
        transition_values = self.get_state_transition_values(self._current_state, real_action)

        next_state = self.make_random_transition(transition_values)

        reward = transition_values.rewards[next_state]
        # if mask_this_step and len(np.where(legal_actions))>1:
        if mask_this_step:
            reward += self.lazy_reward
        difficulty_contribution = transition_values.difficulty_contributions[next_state]
        

        self._cumulative_reward += reward
        self._cumulative_difficulty_contribution += difficulty_contribution
        self._episode_length += 1
        is_done = self._episode_length == self.expected_horizon
        if is_done:
            reward += self._model.get_truncate_reward(next_state)
            # reward += self.episode_mask * self.lazy_reward
            self.episode_mask = 0

        action_name = str(self.action_index_to_action(real_action))
        if action_name not in self._action_counts:
            self._action_counts[action_name] = 0
        self._action_counts[action_name] += 1

        info = {
            'transition':
                f'{self._current_state},{action_name}->{next_state}',
            'reward_ratio': f'{reward}/{difficulty_contribution}',
            'revenue': self.revenue(),
            'length': self._episode_length,
            'actions': dict(sorted((real_action, count / self._episode_length)
                                   for real_action, count in self._action_counts.items())),
        }
        if not self.include_transition_info:
            del info['transition']


        # experience = Experience(prev_state=self.tuple_to_torch(self._current_state), action=action,
        #                         next_state=self.tuple_to_torch(next_state), reward=reward,
        #                         difficulty_contribution=difficulty_contribution,
        #                         prev_difficulty_contribution=self._prev_difficulty_contribution, is_done=is_done,
        #                         legal_actions=legal_actions, target_value=None, info=info)

        if not legal_actions[real_action]:
            return self.tuple_to_torch(self._current_state), self.error_penalty, is_done, info
        self._current_state = next_state
        self._prev_difficulty_contribution = difficulty_contribution
        # return experience
        return self.tuple_to_torch(next_state), reward, is_done, info

    @staticmethod
    def make_random_transition(transition_values: StateTransitions) -> BlockchainModel.State:
        possible_states = list(transition_values.probabilities.keys())
        possible_states_array = np.arange(len(possible_states))
        probabilities = np.array(list(transition_values.probabilities.values()))
        next_state_index = np.random.choice(possible_states_array, p=probabilities)
        next_state = possible_states[next_state_index]
        return next_state

    def get_state_legal_actions_tensor(self, state: Union[BlockchainModel.State, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(state):
            state = self.torch_to_tuple(state)

        legal_actions_list = self.get_state_legal_actions(state)

        legal_actions = torch.zeros((self.num_of_actions,), device=self.device, dtype=torch.bool)
        for action in legal_actions_list:
            legal_actions[action] = 1

        return legal_actions

    def get_state_legal_actions(self, state: BlockchainModel.State) -> List[int]:
        legal_actions = []
        for action_index in range(self.num_of_actions):
            action = self.action_index_to_action(action_index)
            # Illegal by def
            if action is self._model.Action.Illegal or \
                    (isinstance(action, tuple) and action[0] is self._model.Action.Illegal):
                continue

            transitions = self._model.get_state_transitions(state, self.action_index_to_action(action_index),
                                                            check_valid=self.check_valid_states)

            # Illegal if returned error penalty
            if self.final_state in transitions.probabilities \
                    and transitions.probabilities[self.final_state] == 1 \
                    and transitions.rewards[self.final_state] == self._model.error_penalty:
                continue

            legal_actions.append(action_index)

        return legal_actions

    def revenue(self) -> float:
        try:
            return self._cumulative_reward / self._cumulative_difficulty_contribution
        except ZeroDivisionError:
            return 0

    def reset(self, state: Optional[BlockchainModel.State] = None) -> Experience:
        if state is None:
            state = self._model.initial_state

        self._current_state = state
        self._prev_difficulty_contribution = 0
        self._cumulative_reward = 0
        self._cumulative_difficulty_contribution = 0
        self._episode_length = 0
        self._action_counts = {}

        # return Experience(prev_state=None, action=None, next_state=self.tuple_to_torch(self._current_state),
        #                   reward=None, difficulty_contribution=None, prev_difficulty_contribution=None, is_done=None,
        #                   legal_actions=self.get_state_legal_actions_tensor(self._current_state),
        #                   target_value=None, info=None)
        return self.tuple_to_torch(self._current_state)
