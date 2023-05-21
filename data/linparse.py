import os
import re
import json
import copy
import argparse
from functools import cmp_to_key

PLAYERS = {0:'E', 1:'S', 2:'W', 3:'N'}
SUITMAP = {'C':0, 'D':1, 'H':2, 'S':3}
CARDMAP = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}

def rindex(list, elt):
    return len(list) - list[::-1].index(elt) -1
   

class Card(object):
    suit_names = ['C', 'D', 'H', 'S']
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank
        self.suitname = self.suit_names[suit]
        self.rankname = self.rank_names[rank-2]

    def __repr__(self):
        """Returns a human-readable string representation."""
        return '%s%s' % (Card.rank_names[self.rank-2], Card.suit_names[self.suit])
    
    def __eq__(self, other):
        return (self.suit == other.suit and self.rank == other.rank)

    def compare(self, other, sorder=(0,1,2,3)):
        self_srank = sorder.index(self.suit)
        other_srank = sorder.index(other.suit)

        if self==other:
            return 0
        elif (self_srank < other_srank):
            return -1
        elif (self_srank == other_srank) and (self.rank < other.rank):
            return -1
        else:
            return 1

class Hand(object):
    def __init__(self, initial=None):
        if initial is None:
            self.cards = []
        else:
            self.cards = initial

    def __repr__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return ','.join(res)

    def __getitem__(self, item):
        return self.cards[item]

    def __len__(self):
        return len(self.cards)

    def has(self, other):
        return (other in self.cards)

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        self.cards.remove(card)
    
    def pop_card(self, i=-1):
        return self.cards.pop(i)
    
    def sort(self, sorder=(0,1,2,3)):
        keyfunc = cmp_to_key(lambda x,y: x.compare(y, sorder))

        return Hand(initial=sorted(self, key=keyfunc))

class BridgeHand:
    def __init__(self, players, dealer, hands, bids, play, contract, declarer, doubled, vuln, made, claimed):
        self.players = players
        self.dealer = dealer
        self.hands = hands
        self.bids = bids
        self.play = play
        self.contract = contract
        self.declarer = declarer
        self.doubled = doubled
        self.vuln = vuln
        self.made = made
        self.claimed = claimed

    def toDICT(self):
        dct = copy.deepcopy(self.__dict__)
        dct['hands'] = {k: str(v) for k, v in dct['hands'].items()}
        return dct

    # might want to define a method for checking equality here

def full_hand():
    fullhand = Hand()
    for suit in range(4):
        for rank in range(2,15):
            fullhand.cards.append(Card(suit,rank))

    return fullhand

def convert_card(lincard):
    '''
    Convert lin-style 2-char notation to Card()
    '''
    
    linsuit = lincard[0]
    linval = lincard[1]
    return Card(SUITMAP[linsuit], CARDMAP[linval])


def rotate_to(dir, offset=0, list='ESWN'):
    suits = ['E','S','W','N']
    rotation = suits.index(dir)+offset

    return list[rotation:] + list[:rotation]

def get_players(lin):
    p_match = re.search('pn\|([^\|]+)', lin)
    if not(p_match):
        return None
    
    p_str = p_match.group(1).split(',')

    # player order is always S,W,N,E
    players = {'S':p_str[0], 'W':p_str[1], 'N':p_str[2], 'E':p_str[3]}
    return(players)
    
def get_dealer(lin):
    match = re.search(r'md\|([1-4])', lin)
    dealer_no = int(match.group(1))

    # have to %4 since BBO indexes 1-4 while we use 0-3
    return PLAYERS[dealer_no%4]


def get_initial_hands(lin):
    def convert_cards(cstring):
        hand = Hand()

        card_match = re.search('S([^HDC]*)?H([^DC]*)?D([^C]*)?C(.*)?', cstring)
        for s in range(4):
            for v in card_match.group(s+1):
                hand.add_card(Card(suit=(3-s), rank=CARDMAP[v]))

        return hand 


    # get the hands of S, W, N
    hands = {}

    h_match = re.search('md\|([^\|]+)', lin)
    if not(h_match):
        return None

    h_str = h_match.group(1).split(',')

    # strip the dealer flag from the first blob
    h_str[0] = h_str[0][1:]

    hands['S'] = convert_cards(h_str[0])
    hands['W'] = convert_cards(h_str[1])
    hands['N'] = convert_cards(h_str[2])
   
    # recover the hand of E
    hands['E'] = full_hand()
    for dir in 'SWN':
        for card in hands[dir].cards:
            assert card in hands['E'].cards
            hands['E'].remove_card(card)

    return(hands)

def get_bids(lin, dealer):
    BID_PLAYERS = rotate_to(dealer)

    # extract raw bid string 
    bids_match = re.search(r'mb\|(.+?)\|pg', lin)
    if not(bids_match):
        return None

    # extract the bid sequence
    bids_match = bids_match.group(1)
    bidding_end = 'mb|p|mb|p|mb|p'
    idx_end = bids_match.find(bidding_end)
    bids_match = bids_match[:idx_end] + bidding_end
    if idx_end < 0:
        raise Exception("Bid end not found")
    bids = bids_match.split('|mb|')
    bids = [re.sub(r'(\|?an\|(.*)?$)|\!', '', x) for x in bids]
    

    # check for passout
    if (len(bids) == 4) and (bids[0] == 'p'):
        return bids, None, 'PO'
    doubles = []
    i = 1
    while (bids[-i] in 'drp') or (bids[-i] == 'p!'):
        if bids[-i] in 'dr':
            doubles.append(bids[-i])
        i += 1
    contract = bids[-i]
    csuit = contract[1]
    cindex = bids.index(contract)
    
    def get_snd(str):
        if len(str) == 1: return None
        else: return str[1]
    bidsuits = list(map(get_snd, bids))
    firstmatch = rindex(bidsuits[cindex::-2], csuit)
    
    if firstmatch % 2 == 0:
        declarer = BID_PLAYERS[cindex % 4]
    else:
        declarer = BID_PLAYERS[(cindex-2) % 4]

    return bids, declarer, (contract, len(doubles))

def get_vulnerability(lin):
    match = re.search(r'sv\|(.)\|', lin)
    vuln_str = match.group(1) 

    if vuln_str in 'NnSs':
        return 'NS'
    elif vuln_str in 'EeWw':
        return 'WE'
    elif vuln_str in 'oO0':
        return 'none'
    elif vuln_str in 'Bb':
        return 'both'

def parse_linfile(linfile):
    with open(linfile, 'r') as f:
        lin = f.read()
    games = []
    for linn in lin.split('\n'):
        try:
            players = get_players(linn)
            dealer = get_dealer(linn)
            hands = get_initial_hands(linn)
            bids_triple = get_bids(linn, dealer)

            if not(hands and bids_triple):
                return None
            else:
                bids, declarer, (contract, doubled) = bids_triple

            vuln = get_vulnerability(linn)
            game = BridgeHand(players, dealer, hands, bids, None, contract, declarer, doubled, vuln, None, None)
            games.append(game)
        except Exception as exception:
            continue
    return games

def extract_games_from_lin(filename, read_path='expert_data_trimmed', write_path='expert_data_final'):
    root = '.'.join(filename.split('.')[:-1])
    games = parse_linfile(os.path.join(read_path, filename))
    for i, game in enumerate(games):
        with open(f"./{write_path}/{root}_{i}.json", "w") as fp:
            json.dump(game.toDICT(), fp, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parsing line files')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    args = parser.parse_args()
    for i in range(args.start, args.end + 1):
        print(f"extracting games from games{i}")
        try:
            extract_games_from_lin(f'games{i}.lin')
        except Exception:
            continue

