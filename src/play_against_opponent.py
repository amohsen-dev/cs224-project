import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from utils import json_to_lin_cards, calc_score_adj, calc_imp, eval_trick_from_game
from behavioral_cloning_test import ENN, PNN
from utils import generate_random_game, label_to_bid, bid_to_label
from extract_features import extract_from_incomplete_game 

PLAYERS = ['S', 'W', 'N', 'E']

if __name__ == '__main__':
    game1 = generate_random_game()
    game2 = copy.deepcopy(game1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_enn = ENN().to(device)
    model_pnn = PNN().to(device)
    model_enn.load_state_dict( torch.load( '../model_cache/model_372_52_e5/model_enn_19.data' ) )
    model_pnn.load_state_dict( torch.load( '../model_cache/model_pnn/model_pnn_19.data' ) )

    print(json.dumps(game1, indent=4))
    game_scores = []
    for robot_players, game in zip(['EW', 'NS'], [game1, game2]):
        npasses = 0
        bidding_player = []
        current_player = game['dealer']
        current_index = PLAYERS.index(current_player)
        print(f'PNN as {robot_players} playing against oppenent')
        while True:
            if current_player in robot_players:
                x = extract_from_incomplete_game(game)
                enn_input = torch.from_numpy( np.concatenate( x[:3] ) ).type( torch.float32 ) 
                partner_hand_estimation = model_enn( enn_input )
                pnn_input = torch.concat([enn_input, partner_hand_estimation])
                action = model_pnn( pnn_input )
                bid = int(torch.nn.Softmax(dim=0)(action).argmax(dim=0))
                bid = label_to_bid(bid)
                print(f'{current_player} - PNN bids: {bid}')
            else:
                bid = ''
                while len(bid)==0:
                    bid = input(f'{current_player} - Enter opponent bid: ')
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
        
        doubled = 0
        contract = None
        contract_suit = None
        contract_bidder = None
        declarer = None
        if game['bids'] != ['p', 'p', 'p', 'p']:
            for bid, bidder in zip(game['bids'][::-1], bidding_player[::-1]):
                if bid == 'd':
                    doubled = 1
                if bid == 'r':
                    doubled = 2
                if bid not in ['p', 'r', 'd']:
                    contract = bid
                    contract_bidder = 'EW' if bidder in 'EW' else 'NS'
                    break
        if contract is not None:
            contract_suit = contract[-1]
            for bid, bidder in zip(game['bids'], bidding_player):
                if bidder in contract_bidder and contract_suit == bid[-1]:
                    declarer = bidder
                    break
        game['contract'] = contract
        game['doubled'] = doubled
        game['declarer'] = declarer


        print(json.dumps(game, indent=4))
        print('\n')
        print('bidding players: ', bidding_player)
        print('bidding sequence: ', game['bids'])
        trick = eval_trick_from_game(robot_players, game)
        print('Theoretical tricks:')
        kwargs_cs = {
            'pos': robot_players,
            'declarer': game['declarer'],
            'contract': game['contract'],
            'trick': trick,
            'vuln': game['vuln'],
            'doubled': game['doubled']}
        game_score = calc_score_adj(**kwargs_cs)
        game_scores.append(game_score)
        print(ev)

    IMP = calc_imp(sum(game_scores))
    print('*' * 60 + f'  IMP = {IMP}  ' + '*' * 60)
