import os
import asyncio
import numpy as np
import pandas as pd
from compute_score import calc_score, calc_IMP
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
CARD_INDEX = { c + s: CARDNO_INDEX[c] + NUM_CARDS * SUIT_INDEX[s]  for s, c in product(['C', 'D', 'H', 'S'], CARDNO_INDEX.keys())}
DECK = list(CARD_INDEX.keys())

vulns = ['both', 'NS', 'EW', 'none']

def generate_random_hands():
    deck_tmp = DECK.copy()
    hands = {}
    for h in PLAYERS:
        hands[h] = sorted(np.random.choice(deck_tmp, size=13, replace=False).tolist(), key=lambda c: CARD_INDEX[c], reverse=True)
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


def get_info_from_game_and_bidders(game, bidding_players):
    doubled = 0
    contract = None
    contract_bidder = None
    declarer = None
    if game['bids'] == ['p', 'p', 'p', 'p']:
        print('GAME ABORTED ALL PASS')
        return None, None, None
    for bid, bidder in zip(game['bids'][::-1], bidding_players[::-1]):
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
        for bid, bidder in zip(game['bids'], bidding_players):
            if bidder in contract_bidder and contract_suit == bid[-1]:
                declarer = bidder
                break
    return contract, declarer, doubled
def eval_trick_from_game(players, game):
    clin = json_to_lin_cards(game)
    ev = os.popen(f'../solver/bcalconsole -e e -q -t a -d lin -c {clin}').read()
    ev = [e.split() for e in ev.split('\n')][:-1]
    ev = pd.DataFrame(ev, columns=['leader', 'C', 'D', 'H', 'S', 'N']).set_index('leader').astype(np.int32)
    ev.index = ev.index.map({'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'})
    trick = ev.loc[game['declarer'], game['contract'][-1]]
    if players == 'EW':
        trick = 13 - trick
    return trick

async def eval_trick_from_game_async(players, game):
    clin = json_to_lin_cards(game)
    cmd = f'../solver/bcalconsole -e e -q -t a -d lin -c {clin}'
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()
    return stdout
    ev = [e.split() for e in stdout.split('\n')][:-1]
    ev = pd.DataFrame(ev, columns=['leader', 'C', 'D', 'H', 'S', 'N']).set_index('leader').astype(np.int32)
    ev.index = ev.index.map({'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'})
    trick = ev.loc[game['declarer'], game['contract'][-1]]
    if players == 'EW':
        trick = 13 - trick
    return trick

def calc_score_adj(pos, declarer, contract, trick, vuln, doubled, verbose=False):
    if doubled == 'r':
        doubled, redoubled = 0, 1
    elif doubled == 'd':
        doubled, redoubled = 1, 0
    else:
        doubled, redoubled = 0, 0

    vul = 0
    if vuln == 'both' or (pos in vuln):
        vul = 1

    level, suit = int(contract[:-1]), SUIT_INDEX[contract[-1]]

    if verbose:
        print(f'calling calc_score with args ({level, suit, trick, vul, doubled, redoubled})')
    score = calc_score(level, suit, trick, vul, doubled, redoubled)
    if verbose:
        print(f'received score {score}')
    if declarer in 'EW':
        score = -score
    if ((pos in 'EW') and (declarer in 'NS')) or ((pos in 'NS') or (declarer in 'EW')):
        score = -score
    if verbose:
        print(f'score adjusted to {score}')
    return score


def calc_imp(x):
    return calc_IMP(x)


if __name__ == '__main__':
    l = bid_to_label('3H')
    lb = label_to_bid(l)
    print(lb)
