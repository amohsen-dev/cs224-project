import os
import json
import copy
import torch
import asyncio
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from utils import ENN, PNN, BaselineNet, calc_score_adj, calc_imp, eval_trick_from_game, get_info_from_game_and_bidders, eval_trick_from_game_async
from utils import generate_random_game, label_to_bid, bid_to_label, MAX_ITER
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
        with torch.no_grad():
            x = extract_from_incomplete_game(game)
            state = torch.from_numpy(np.concatenate(x[:3])).type(torch.float32)
            partner_hand_estimation = self.model_enn(state)
            pnn_input = torch.concat([state, partner_hand_estimation])
            pnn_input.requires_grad = False
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
    try:
        game1 = generate_random_game()
        game2 = copy.deepcopy(game1)

        if verbose:
            print(json.dumps(game1, indent=4))
        game_scores = []
        path = {'states': [], 'actions': [], 'rewards': []}
        for agent1_side, game in zip(['EW', 'NS'], [game1, game2]):
            npasses = 0
            bidding_player = []
            current_player = game['dealer']
            current_index = PLAYERS.index(current_player)
            if verbose:
                print(f'PNN as {agent1_side} playing against oppenent')
            it = 0
            while True:
                if current_player in agent1_side:
                    state, bid = agent1.bid(game)
                    path['states'].append(state)
                    action = bid_to_label(bid)
                    path['actions'].append(torch.Tensor([action]).type(torch.int32))
                    path['rewards'].append(0)
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
                it += 1
                if it > MAX_ITER:
                    print(game['bids'])
                    raise Exception('MAX ITER')

            contract, declarer, doubled = get_info_from_game_and_bidders(game, bidding_player)
            if contract is None:
                break
            game['contract'] = contract
            game['doubled'] = doubled
            game['declarer'] = declarer

            if verbose:
                print(json.dumps(game, indent=4))
            #trick = eval_trick_from_game(agent1_side, game)
            trick = asyncio.run(eval_trick_from_game_async(agent1_side, game))
            game_score = calc_score_adj(agent1_side, game['declarer'], game['contract'], trick, game['vuln'], game['doubled'], verbose)
            game_scores.append(game_score)

        IMP = calc_imp(sum(game_scores))
        path['rewards'][-1] = IMP
        if verbose:
            print('*' * 60 + f'  IMP = {IMP}  ' + '*' * 60)
        if contract is None:
            path = None
        return path
    except Exception as exception:
        raise(exception)
        return None

class PolicyGradient:
    def __init__(self):
        self.agent_target = PNNAgent()
        self.agent_opponent = PNNAgent()
        self.baseline_net = BaselineNet()
        self.num_episodes = 8
        self.num_epochs = 8
        self.gamma = 0.99
        self.ppo_epsilon = 1e-1
        self.enn_opt = torch.optim.Adam(self.agent_target.model_enn.parameters(), lr=1e-3)
        self.pnn_opt = torch.optim.Adam(self.agent_target.model_pnn.parameters(), lr=1e-3)
        self.baseline_opt = torch.optim.Adam(self.baseline_net.parameters(), lr=1e-3)
    def generate_paths(self):
        paths = []
        for _ in tqdm(range(self.num_episodes)):
        #for _ in range(self.num_episodes):
            path = play_random_game(self.agent_target, self.agent_opponent, verbose=False)
            if path is not None:
                paths.append(path)
        return paths

    def update_policy(self, paths, PPO=False):
        old_log_probs = []
        for path in paths:
            this_old_log_probs = []
            for state, action in zip(path['states'], path['actions']):
                enn = self.agent_target.model_enn(state)
                logits = self.agent_target.model_pnn(torch.cat([state, enn]))
                dist = Categorical(logits=logits)
                this_old_log_probs.append(dist.log_prob(action).detach())
            old_log_probs.append(this_old_log_probs)

        losses = []
        for epoch in range(self.num_epochs):
            self.enn_opt.zero_grad()
            self.pnn_opt.zero_grad()
            loss = torch.Tensor([0]).type(torch.float32)
            N = 0
            for path, old_log_probss in zip(paths, old_log_probs):
                path_len = len(path['rewards'])
                returns = np.power(self.gamma, np.arange(path_len)[::-1]) * path['rewards'][-1]
                for state, action, A, old_log_prob in zip(path['states'], path['actions'], returns, old_log_probss):
                    enn = self.agent_target.model_enn(state)
                    logits = self.agent_target.model_pnn(torch.cat([state, enn]))
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(action)
                    if PPO:
                        ratio = torch.exp(log_prob - old_log_prob)
                        ratio_clipped = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
                        loss += torch.min(ratio * A, ratio_clipped * A)
                    else:
                        loss += - log_prob * A
                    N += 1
            loss = loss / N
            loss.backward()
            self.enn_opt.step()
            self.pnn_opt.step()
            losses.append(loss.detach().numpy()[0])
        print(losses)
        imp = np.mean([p['rewards'][-1] for p in paths])
        return imp


if __name__ == '__main__':

    test = {'players': None, 'dealer': 'W', 'hands': {'S': '6S,5S,10H,6H,KD,6D,5D,3D,2D,AC,JC,10C,6C', 'W': 'AS,10S,3S,KH,QH,JH,5H,4H,3H,JD,9D,KC,3C', 'N': 'KS,QS,7S,AH,8H,2H,QD,8D,9C,8C,7C,5C,2C', 'E': 'JS,9S,8S,4S,2S,9H,7H,AD,10D,7D,4D,QC,4C'}, 'bids': ['1H', '2C', '2H', 'p', '3H', 'd', '3S', 'p', '3N', '4H', 'p', 'p', 'd', 'p', '4S', 'p', 'p', 'd', 'd', 'p', 'p', '5C', 'd', 'd', 'd', 'd', 'd', 'p', 'd', 'd', 'd', 'p', 'p'], 'play': None, 'contract': None, 'declarer': None, 'doubled': None, 'vuln': 'both', 'made': None, 'claimed': None}
    extract_from_incomplete_game(test)


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--ppo', action='store_true')
    args = parser.parse_args()

    algorithm = PolicyGradient()
    imps, losses = [], []
    for i in range(1000):
        paths = algorithm.generate_paths()
        if len(paths) > 0:
            imp = algorithm.update_policy(paths, PPO=args.ppo)
            imps.append(imp)
        print('IMP MEAN:')
        print(np.cumsum(imps) / (1 + np.arange(len(imps))))

    print('path generated')