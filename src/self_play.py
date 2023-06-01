import torch
import numpy as np
from behavioral_cloning_test import ENN, PNN
from utils import generate_random_game
from extract_features import extract_from_incomplete_game 


if __name__ == '__main__':
    game = generate_random_game()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_enn = ENN().to(device)
    model_pnn = PNN().to(device)
    model_enn.load_state_dict( torch.load( 'model_cache/model_372_52_e5/model_enn_19.data' ) )
    # model_pnn.load_state_dict( torch.load( 'model_cache/model_pnn/model_pnn_19.data' ) )
    pass
    x = extract_from_incomplete_game(game)
    enn_input = torch.from_numpy( np.concatenate( x[:3] ) ).type( torch.float32 ) 
    partner_hand_estimation = model_enn( enn_input )
    pnn_input = torch.concat([enn_input, partner_hand_estimation])
    action = model_pnn( pnn_input )
    pass
