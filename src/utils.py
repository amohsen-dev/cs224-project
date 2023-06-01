import json
import numpy as np
from itertools import product

NUM_PLAYERS = 4
DECK_SIZE = 52
NUM_SUITS = 4
NUM_CARDS = 13
MAX_BID_LENGTH = 318
NUM_BIDS = 38


PLAYERS = ['S', 'W', 'N', 'E']
PLAYER_INDEX = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
tmp = np.arange(2, 11).astype(np.int32)
CARDNO_INDEX = dict(zip(map(str, tmp), tmp-2))
CARDNO_INDEX = {**CARDNO_INDEX, **{'J': 9, 'Q': 10, 'K': 11, 'A': 12}}
SUIT_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3, 'N': 4}
SUIT_INDEX_REV = {v: k for k, v in SUIT_INDEX.items()}
CARD_INDEX = { c + s: CARDNO_INDEX[c] + NUM_CARDS * SUIT_INDEX[s]  for c, s in product(CARDNO_INDEX.keys(), ['C', 'D', 'H', 'S'])}
DECK = list(CARD_INDEX.keys())

vulns = ['both', 'NE', 'SW', 'none']

def generate_random_hands():
    deck_tmp = DECK.copy()
    hands = {}
    for h in PLAYERS:
        hands[h] = np.random.choice(deck_tmp, size=13, replace=False).tolist()
        deck_tmp = [c for c in deck_tmp if c not in hands[h]]
    return hands

def generate_random_game():
    hands = generate_random_hands()
    hands = {k: ','.join(v) for k, v in hands.items()}
    game = {
            'players': None,
            'dealer': np.random.choice(PLAYERS, 1)[0],
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

def bid_to_label(bid):
    if bid == 'p':
        next_bid = 0
    elif bid[0] == 'd':
        next_bid = 1
    elif bid[0] == 'r':
        next_bid = 2
    else:
        tricks = int(bid[0])
        suit = SUIT_INDEX[bid[1]]
        next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
    return next_bid

def label_to_bid(label):
    if label == 0:
        return 'p'
    elif label == 1:
        return 'd'
    elif label == 2:
        return 'r'
    else:
        tmp = label - 3
        suit = tmp % (1 + NUM_SUITS)
        tricks = 1 + tmp // (1 + NUM_SUITS)
        return str(tricks) + SUIT_INDEX_REV[suit] 

def json_to_lin_cards(dct):
    H = []
    for P in ['S', 'W', 'N', 'E']:
        hh = {'S': [], 'H': [], 'D': [], 'C': []}
        for h in dct['hands'][P].split(','):
            hh[h[-1]].append(h[0] if len(h) == 2 else 'T')
        hhh = ''
        for s in ['S', 'H', 'D', 'C']:
            hhh += s + ''.join(hh[s])
        H.append(hhh)
    return ','.join(H)

if __name__ == '__main__':
    l = bid_to_label('3H')
    lb = label_to_bid(l)
    print(lb)
