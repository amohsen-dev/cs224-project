import os
import json
import copy
import torch
import asyncio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from utils import ENN, PNN, PNN2, BaselineNet, calc_score_adj, calc_imp, get_info_from_game_and_bidders, eval_trick_from_game_async
from utils import generate_random_game, label_to_bid, bid_to_label, MAX_ITER, MaxIterException, BridgeRuleViolation
from extract_features import extract_from_incomplete_game
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

PLAYERS = ['S', 'W', 'N', 'E']

class Agent:
    @abstractmethod
    def bid(self, game):
        NotImplementedError()

class PNNAgent(Agent):
    def __init__(self,
                 path_enn='../model_cache/model_372_52_e5/model_enn_19.data',
                 path_pnn='../model_cache/model_pnn/model_pnn_19.data',
                 pnn2=False,
                 stochastic=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_enn = ENN().to(self.device)
        self.model_pnn = PNN().to(self.device) if not pnn2 else PNN2().to(self.device)
        self.model_enn.load_state_dict(torch.load(path_enn))
        self.model_pnn.load_state_dict(torch.load(path_pnn))
        self.stochastic = stochastic

    def bid(self, game):
        with torch.no_grad():
            x = extract_from_incomplete_game(game)
            state = torch.from_numpy(np.concatenate(x[:3])).type(torch.float32)
            partner_hand_estimation = self.model_enn(state)
            pnn_input = torch.concat([state, partner_hand_estimation])
            pnn_input.requires_grad = False
            logits = self.model_pnn(pnn_input)
            if self.stochastic:
                stoch_policy = Categorical(logits=logits)
                action = stoch_policy.sample()
            else:
                action = torch.argmax(torch.nn.Softmax(dim=0)(logits))
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
    path = {'states': [], 'actions': [], 'rewards': []}
    contract = None
    violation = False
    target_violation = False
    try:
        for agent1_side, game in zip(['EW', 'NS'], [game1, game2]):
            npasses = 0
            bidding_player = []
            current_player = game['dealer']
            current_index = PLAYERS.index(current_player)
            if verbose:
                print(f'PNN as {agent1_side} playing against opponent')
            it = 0
            while True:
                test_contract, _, test_doubled = get_info_from_game_and_bidders(game, bidding_player)

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

                violation |= (test_doubled == 1 and bid == 'd') or (test_doubled == 2 and bid in ['d', 'r'])
                if test_contract is not None and bid not in ['p', 'd', 'r']:
                    violation |= (int(bid[:-1]) < int(test_contract[:-1]))
                elif test_contract is None:
                    violation |= bid in ['d', 'r']
                if violation:
                    if current_player in agent1_side:
                        target_violation = True
                    #  print(bid, ' after ', game['bids'], f' Target vioation: {target_violation}')
                    raise BridgeRuleViolation()

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
                    raise MaxIterException(f'MAX ITER: {json.dumps(game, indent=4)}')

            contract, declarer, doubled = get_info_from_game_and_bidders(game, bidding_player)
            if contract is None:
                break
            game['contract'] = contract
            game['doubled'] = doubled
            game['declarer'] = declarer

            if verbose:
                print(json.dumps(game, indent=4))
            #trick = eval_trick_from_game(agent1_side, game)
            trick = asyncio.run(eval_trick_from_game_async(game, verbose))
            game_score = calc_score_adj(agent1_side, game['declarer'], game['contract'], trick, game['vuln'], game['doubled'], verbose)
            game_scores.append(game_score)
        IMP = calc_imp(sum(game_scores))
    except BridgeRuleViolation:
        if target_violation:
            IMP = -24
        else:
            raise BridgeRuleViolation()

    path['rewards'][-1] = IMP
    if verbose:
        print('*' * 60 + f'  IMP = {IMP}  ' + '*' * 60)
    if contract is None and not target_violation:
        path = None
    return path

class PolicyGradient:
    def __init__(self):
        self.agent_target = PNNAgent()
        self.agent_opponent = PNNAgent()
        self.baseline_net = BaselineNet()
        self.num_episodes = 32
        self.num_epochs = 8
        self.gamma = 0.99
        self.ppo_epsilon = 1e-1
        self.enn_opt = torch.optim.SGD(self.agent_target.model_enn.parameters(), lr=1e-5)
        self.pnn_opt = torch.optim.SGD(self.agent_target.model_pnn.parameters(), lr=1e-4)
        self.baseline_opt = torch.optim.SGD(self.baseline_net.parameters(), lr=1e-2)
    def generate_paths(self):
        paths = []
        for _ in tqdm(range(self.num_episodes)):
            try:
                path = play_random_game(self.agent_target, self.agent_opponent, verbose=False)
                if path is not None:
                    paths.append(path)
            except (MaxIterException, BridgeRuleViolation) as exception:
                print(exception)
                continue
        return paths

    def update_policy(self, paths, PPO=False, Baseline=False):
        old_log_probs = []
        for path in paths:
            this_old_log_probs = []
            for state, action in zip(path['states'], path['actions']):
                enn = self.agent_target.model_enn(state)
                logits = self.agent_target.model_pnn(torch.cat([state, enn]))
                dist = Categorical(logits=logits)
                this_old_log_probs.append(dist.log_prob(action).detach())
            old_log_probs.append(this_old_log_probs)

        if PPO:
            self.num_epochs = 10
        else:
            self.num_epochs = 1
        losses, losses_bl = [], []
        for epoch in range(self.num_epochs):
            self.enn_opt.zero_grad()
            self.pnn_opt.zero_grad()
            self.baseline_opt.zero_grad()
            loss = torch.Tensor([0]).type(torch.float32)
            loss_bl = torch.Tensor([0]).type(torch.float32)
            N, N_bl = 0, 0
            for path, old_log_probss in zip(paths, old_log_probs):
                path_len = len(path['rewards'])
                returns = np.power(self.gamma, np.arange(path_len)[::-1]) * path['rewards'][-1]
                for state, action, ret, old_log_prob in zip(path['states'], path['actions'], returns, old_log_probss):

                    enn = self.agent_target.model_enn(state)
                    if Baseline:
                        v = self.baseline_net(torch.cat([state, enn.detach()]))
                        loss_bl += (v - ret) ** 2
                        A = (ret - v).detach()
                    else:
                        A = ret
                    logits = self.agent_target.model_pnn(torch.cat([state, enn]))
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(action)
                    if PPO:
                        ratio = torch.exp(log_prob - old_log_prob)
                        ratio_clipped = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
                        loss += - torch.min(ratio * A, ratio_clipped * A)
                    else:
                        loss += - log_prob * A
                    N += 1
            loss = loss / N
            loss.backward()
            self.enn_opt.step()
            self.pnn_opt.step()
            if Baseline:
                loss_bl = loss_bl / N
                loss_bl.backward()
                self.baseline_opt.step()
            losses.append(loss.detach().numpy()[0])
            losses_bl.append(loss_bl.detach().numpy()[0])
        print('PNN Losses: ', losses)
        print('Baseline Losses: ', losses_bl)
        imp = np.mean([p['rewards'][-1] for p in paths])
        return imp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--ppo', action='store_true')
    parser.add_argument('-b', '--baseline', action='store_true')
    args = parser.parse_args()

    algorithm = PolicyGradient()
    opponent_pool = [algorithm.agent_opponent]
    imps, losses = [], []
    ff = 'PPO' if args.ppo else 'PG'
    if args.baseline:
        ff += '_AC'
    writer = SummaryWriter(log_dir=f'../model_cache3/RL/{ff}')
    for i in range(224):
        if i % 4 == 0:
            opponent_pool.append(copy.deepcopy(algorithm.agent_target))
            algorithm.agent_opponent = opponent_pool[np.random.choice(len(opponent_pool))]
        if (i % 4 == 0) and (i % 16 != 0):

            torch.save(algorithm.agent_target.model_enn.state_dict(), f"../model_cache3/RL/{ff}/model_enn_{i*algorithm.num_episodes}.data")
            torch.save(algorithm.agent_target.model_pnn.state_dict(), f"../model_cache3/RL/{ff}/model_pnn_{i*algorithm.num_episodes}.data")
        paths = algorithm.generate_paths()
        if len(paths) > 0:
            imp = algorithm.update_policy(paths, PPO=args.ppo, Baseline=args.baseline)
            imps.append(imp)
            writer.add_scalar("Loss/train", imp, i)
        print('IMP MEAN:')
        print(pd.Series(imps).ewm(5).mean())

    print('path generated')
