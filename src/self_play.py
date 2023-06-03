import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from abc import abstractmethod
from utils import json_to_lin_cards, calc_score_adj, calc_imp, eval_trick_from_game, get_info_from_game_and_bidders
from behavioral_cloning_test import ENN, PNN
from utils import generate_random_game, label_to_bid, bid_to_label
from extract_features import extract_from_incomplete_game
from torch.distributions.categorical import Categorical

PLAYERS = ['S', 'W', 'N', 'E']

class Agent:
    @abstractmethod
    def bid(self, game):
        NotImplementedError()

class PNNAgent(Agent):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_enn = ENN().to(self.device)
        self.model_pnn = PNN().to(self.device)
        self.model_enn.load_state_dict(torch.load('../model_cache/model_372_52_e5/model_enn_19.data'))
        self.model_pnn.load_state_dict(torch.load('../model_cache/model_pnn/model_pnn_19.data'))

    def bid(self, game):
        x = extract_from_incomplete_game(game)
        state = torch.from_numpy(np.concatenate(x[:3])).type(torch.float32)
        partner_hand_estimation = self.model_enn(state)
        pnn_input = torch.concat([state, partner_hand_estimation])
        logits = self.model_pnn(pnn_input)
        stoch_policy = Categorical(logits=logits)
        action = stoch_policy.sample()
        #log_prob = stoch_policy.log_prob(action)
        bid = label_to_bid(int(action))
        return state, bid

class ConsoleAgent(Agent):
    def bid(self, game):
        bid = ''
        while len(bid) == 0:
            bid = input(f'{self.__repr__()} - Enter opponent bid: ')
        return None, bid

def play_random_game(agent1: Agent, agent2: Agent, verbose=False):
    game1 = generate_random_game()
    game2 = copy.deepcopy(game1)

    if verbose:
        print(json.dumps(game1, indent=4))
    game_scores = []
    path = []
    for agent1_side, game in zip(['EW', 'NS'], [game1, game2]):
        npasses = 0
        bidding_player = []
        current_player = game['dealer']
        current_index = PLAYERS.index(current_player)
        if verbose:
            print(f'PNN as {agent1_side} playing against oppenent')
        while True:
            if current_player in agent1_side:
                state, bid = agent1.bid(game)
                path.append((state, bid_to_label(bid), 0))
                if verbose:
                    print(f'{current_player} - agent1 bids: {bid}')
            else:
                _, bid = agent2.bid(game)
            if bid == 'p' and (npasses < 2 or len(game['bids']) == 2):
                npasses += 1
            elif bid == 'p':
                game['bids'].append(bid)
                bidding_player.append(current_player)
                break
            else:
                npasses = 0
            game['bids'].append(bid)
            bidding_player.append(current_player)
            current_index = (1 + current_index) % 4
            current_player = PLAYERS[current_index]

        contract, declarer, doubled = get_info_from_game_and_bidders(game, bidding_player)
        if contract is None:
            break
        game['contract'] = contract
        game['doubled'] = doubled
        game['declarer'] = declarer

        if verbose:
            print(json.dumps(game, indent=4))
        trick = eval_trick_from_game(agent1_side, game)
        game_score = calc_score_adj(agent1_side, game['declarer'], game['contract'], trick, game['vuln'], game['doubled'], verbose)
        game_scores.append(game_score)

    IMP = calc_imp(sum(game_scores))
    path[-1] = (path[-1][0], path[-1][1], IMP)
    if verbose:
        print('*' * 60 + f'  IMP = {IMP}  ' + '*' * 60)
    if contract is None:
        path = None
    return path


if __name__ == '__main__':
    agent1 = PNNAgent()
    agent2 = PNNAgent()
    num_paths = 1000
    paths = []
    for i in range(num_paths):
        try:
            path = play_random_game(agent1, agent2, verbose=False)
            if path is not None:
                paths.append(path)
        except Exception as exception:
            print(exception)
    print('path generated')