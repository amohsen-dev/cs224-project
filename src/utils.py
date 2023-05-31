import json
import numpy as np
from itertools import product
from extract_features_new import extract


deck = [''.join(c) for c in product(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'], ['H', 'S', 'D', 'C'])]
dealers = ['S', 'W', 'N', 'E']
vulns = ['both', 'NE', 'SW', 'none']

def generate_random_hands():
    deck_tmp = deck.copy()
    hands = {}
    for h in dealers:
        hands[h] = np.random.choice(deck_tmp, size=13, replace=False).tolist()
        deck_tmp = [c for c in deck_tmp if c not in hands[h]]
    return hands

def generate_random_game():
    hands = generate_random_hands()
    hands = {k: ','.join(v) for k, v in hands.items()}
    game = {
            'players': None,
            'dealer': np.random.choice(dealers, 1)[0],
            'hands': hands,
            'bids': [],
            'play': None,
            'contract': None,
            'declarer': None,
            'doubled': None,
            'vuln': np.random.choice(vulns, 1)[0],
            'made': None,
            'claimed': None
            }


    return game

if __name__ == '__main__':
    with open('tmp/random_game.json', 'w') as fp:
        json.dump(generate_random_game(), fp, indent=4)
    from extract_features import extract2
    data2 = extract2('data/expert_data_final/games1000_0.json')
    data = extract('data/expert_data_final/games1000_0.json')
    print(data)
