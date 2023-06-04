# -*- coding: utf-8 -*-
"""
Created on Mon May 22 02:02:04 2023

@author: samga
"""
import numpy as np
import argparse
import re
import json
from itertools import product
from utils import MAX_ITER, MaxIterException

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
CARD_INDEX = { c + s: CARDNO_INDEX[c] + NUM_CARDS * SUIT_INDEX[s]  for c, s in product(CARDNO_INDEX.keys(), SUIT_INDEX.keys())} 
DECK = CARD_INDEX.keys()


def hand_encoding(hand_str):
    cards = hand_str.split(',')
    hand = np.zeros(DECK_SIZE, dtype=np.int32)
    for c in cards:
        hand[CARD_INDEX[c]] = 1
    return hand
    

def vul_encoding(vul_char, player):
    if vul_char == 'n':
        vul = np.array([1,1])
    elif vul_char == 'b':
        vul = np.array([0,0])
    elif (vul_char == 'N'):
        if player == 0 or player == 2:
            vul = np.array([1,0])
        else:
            vul = np.array([0,1])          
    else:
        if player == 1 or player == 3:
            vul = np.array([1,0])
        else:
            vul = np.array([0,1])          
    return vul

def extract_from_file(input_file):
    with open(input_file, 'r') as fp:
        DICT = json.load(fp)
    return extract_from_dict(DICT)


def extract_from_dict(DICT):
    data = []
    vul_char = DICT['vuln'][0]
    dealer_char = DICT['dealer']
    dealer = PLAYER_INDEX[dealer_char]
    
    hands = np.zeros((NUM_PLAYERS,DECK_SIZE), dtype=int)
    player = -1
    for i in range(NUM_PLAYERS):
        next_player = PLAYER_INDEX[PLAYERS[i]]
        hand_str = DICT['hands'][PLAYERS[i]]
        hands[i,:] = hand_encoding(hand_str)
        if next_player == dealer:
            player = i
    next_hand = hands[player,:]
    partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
    next_vul = vul_encoding(vul_char, player)
    bid_sequence = np.zeros((MAX_BID_LENGTH), dtype=int)
    bid_list = DICT['bids']
    bid_id = 0
    bid = bid_list[bid_id]
    state = 0
    if bid == 'p':
        next_bid = 0
    else:
        tricks = int(bid[0])
        suit = SUIT_INDEX[bid[1]]
        next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
        bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
        state = -1
    data.append((next_hand, next_vul, np.copy(bid_sequence), np.array([next_bid]), partner_hand))
    player = (player + 1)%NUM_PLAYERS
    next_hand = hands[player,:]
    partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
    next_vul = vul_encoding(vul_char, player)
    num_initial_passes = 0
    it = 0
    while state != -1:
        bid_sequence[num_initial_passes] = 1
        num_initial_passes = num_initial_passes + 1
        if num_initial_passes == 4:
            return data
        bid_id += 1
        bid = bid_list[bid_id]
        if bid == 'p':
            next_bid = 0
        else:
            tricks = int(bid[0])
            suit = SUIT_INDEX[bid[1]]
            next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
            bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
            state = -1
        data.append((next_hand, next_vul, np.copy(bid_sequence), np.array([next_bid]), partner_hand))
        player = (player + 1)%NUM_PLAYERS
        next_hand = hands[player,:]
        partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
        next_vul = vul_encoding(vul_char, player)
        it += 1
        if it > MAX_ITER:
            raise MaxIterException('MAX ITER')
    it = 0
    while True:
        state = 0
        bid_sequence[bid_index] = 1
        bid_id += 1
        bid = bid_list[bid_id]
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
            bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
            state = -1
        data.append((next_hand, next_vul, np.copy(bid_sequence), np.array([next_bid]), partner_hand))
        player = (player + 1)%NUM_PLAYERS
        next_hand = hands[player,:]
        partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
        next_vul = vul_encoding(vul_char, player)
        num_passes_1 = 0
        num_passes_2 = 0
        num_passes_3 = 0
        num_doubles = 0
        num_redoubles = 0

        it2 = 0
        while state != -1:
            if bid[0] == 'd':
                bid_sequence[bid_index + 3] = 1
                num_doubles = 1
            elif bid[0] == 'r':
                bid_sequence[bid_index + 6] = 1
                num_redoubles = 1
            else:
                if num_doubles == 0:
                    bid_sequence[bid_index + num_passes_1 + 1] = 1
                    num_passes_1 = num_passes_1 + 1
                    if num_passes_1 == 3:
                        return data
                else:
                    if num_redoubles == 0:
                        bid_sequence[bid_index + num_passes_2 + 4] = 1
                        num_passes_2 = num_passes_2 + 1
                        if num_passes_2 == 3:
                            return data
                    else:
                        bid_sequence[bid_index + num_passes_3 + 7] = 1
                        num_passes_3 = num_passes_3 + 1
                        if num_passes_3 == 3:
                            return data
            bid_id += 1
            bid = bid_list[bid_id]
            if bid[0] == 'p':
                next_bid = 0
            elif bid[0] == 'd':
                next_bid = 1
            elif bid[0] == 'r':
                next_bid = 2
            else:
                tricks = int(bid[0])
                suit = SUIT_INDEX[bid[1]]
                next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
                bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
                state = -1
            data.append((next_hand, next_vul, np.copy(bid_sequence), np.array([next_bid]), partner_hand))
            player = (player + 1)%NUM_PLAYERS
            next_hand = hands[player,:]
            partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
            next_vul = vul_encoding(vul_char, player)

            it2 += 1
            if it2 > MAX_ITER:
                raise MaxIterException('MAX ITER')

        it += 1
        if it > MAX_ITER:
            raise MaxIterException('MAX ITER')
    

def extract_from_incomplete_game(DICT):
    DICT_with_termination = DICT.copy()
    DICT_with_termination['bids'] = DICT_with_termination['bids'] + ['p'] * 4
    idx = len(DICT['bids'])
    extracted = extract_from_dict(DICT_with_termination)
    idx = min(idx, len(extracted) - 1)
    return extracted[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract features')
    parser.add_argument('--input_file', type=str, default='',
                        help='input file')
    parser.add_argument('--output_file', type=str, default='',
                        help='output file')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    data = extract(input_file)
    
    f = open(output_file, "a")
    for i in range(len(data)):
        next_hand, next_vul, bid_sequence, next_bid, partner_hand = data[i]
        for j in range(len(next_hand)):
            f.write(str(next_hand[j]))
        f.write("\n")
        for j in range(len(next_vul)):
            f.write(str(next_vul[j]))
        f.write("\n")
        for j in range(len(bid_sequence)):
            f.write(str(bid_sequence[j]))
        f.write("\n")
        f.write(str(next_bid))
        f.write("\n")
        for j in range(len(partner_hand)):
            f.write(str(partner_hand[j]))
        f.write("\n")
    f.close() 
