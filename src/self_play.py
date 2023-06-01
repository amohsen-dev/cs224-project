import torch
import numpy as np
from behavioral_cloning_test import ENN, PNN
from utils import generate_random_game, label_to_bid, bid_to_label
from extract_features import extract_from_incomplete_game 


if __name__ == '__main__':
    game = generate_random_game()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_enn = ENN().to(device)
    model_pnn = PNN().to(device)
    model_enn.load_state_dict( torch.load( 'model_cache/model_372_52_e5/model_enn_19.data' ) )
    model_pnn.load_state_dict( torch.load( 'model_cache/model_pnn/model_pnn_19.data' ) )

    model_enn_opp = ENN().to(device)
    model_pnn_opp = PNN().to(device)
    model_enn_opp.load_state_dict( torch.load( 'model_cache/model_372_52_e5/model_enn_19.data' ) )
    model_pnn_opp.load_state_dict( torch.load( 'model_cache/model_pnn/model_pnn_19.data' ) )

    npasses = 0
    while True:
        x = extract_from_incomplete_game(game)
        enn_input = torch.from_numpy( np.concatenate( x[:3] ) ).type( torch.float32 ) 
        partner_hand_estimation = model_enn( enn_input )
        pnn_input = torch.concat([enn_input, partner_hand_estimation])
        action = model_pnn( pnn_input )
        bid = int(torch.nn.Softmax()(action).argmax())
        bid = label_to_bid(bid)
        if bid == 'p' and npasses < 2:
            npasses += 1
        elif bid == 'p':
            print(game['bids'] + [bid])
            break
        else:
            npasses = 0
        game['bids'].append(bid)
