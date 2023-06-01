import os
import json
import torch
import numpy as np
from utils import json_to_lin_cards
from behavioral_cloning_test import ENN, PNN
from utils import generate_random_game, label_to_bid, bid_to_label
from extract_features import extract_from_incomplete_game 

PLAYERS = ['S', 'W', 'N', 'E']

if __name__ == '__main__':
    game = generate_random_game()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_enn = ENN().to(device)
    model_pnn = PNN().to(device)
    model_enn.load_state_dict( torch.load( 'model_cache/model_372_52_e5/model_enn_19.data' ) )
    model_pnn.load_state_dict( torch.load( 'model_cache/model_pnn/model_pnn_19.data' ) )

    npasses = 0
    current_player = game['dealer']
    current_index = PLAYERS.index(current_player)
    robot_players = input('Which side is PNN robot on ? choose EW/NS: ')
    print(json.dumps(game, indent=4))
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
            bid = input(f'{current_player} - Enter opponent bid: ')
        if bid == 'p' and npasses < 2:
            npasses += 1
        elif bid == 'p':
            game['bids'].append(bid)
            break
        else:
            npasses = 0
        game['bids'].append(bid)
        current_index = (1 + current_index) % 4
        current_player = PLAYERS[current_index]

    print('\n')
    input('Are you ready to see the results?')
    print('bids during the game: ', game['bids'])
    clin = json_to_lin_cards(game)
    ev = os.popen(f'solver/bcalconsole -e e -q -t a -d lin -c {clin}').read()
    print('Theoretical tricks:')
    print(ev)
