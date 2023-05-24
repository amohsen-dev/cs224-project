# -*- coding: utf-8 -*-
"""
Created on Mon May 22 02:02:04 2023

@author: samga
"""
import numpy as np
import argparse
import re

NUM_PLAYERS = 4
DECK_SIZE = 52
NUM_SUITS = 4
NUM_CARDS = 13
MAX_BID_LENGTH = 318
NUM_BIDS = 38

def player_index(player_char):
    if player_char == 'N':
        index = 0
    elif player_char == 'E':
        index = 1
    elif player_char == 'S':
        index = 2
    else:
        index = 3
    return index

def card_index(card_char):
    if card_char == '2':
        index = 0
    elif card_char == '3':
        index = 1
    elif card_char == '4':
        index = 2
    elif card_char == '5':
        index = 3
    elif card_char == '6':
         index = 4
    elif card_char == '7':
         index = 5
    elif card_char == '8':
         index = 6
    elif card_char == '9':
         index = 7
    elif card_char == '1':
         index = 8
    elif card_char == 'J':
         index = 9
    elif card_char == 'Q':
         index = 10
    elif card_char == 'K':
         index = 11
    else:
        index = 12
    return index

def suit_index(suit_char):
    if suit_char == 'C':
        index = 0
    elif suit_char == 'D':
        index = 1
    elif suit_char == 'H':
        index = 2
    elif suit_char == 'S':
        index = 3
    else:
        index = 4
    return index

def hand_encoding(line):
    player_char = line[1]
    player = player_index(player_char)
    hand = np.zeros((DECK_SIZE,), dtype=int)
    start_index = 5
    for i in range(NUM_CARDS):
        card_char = line[start_index]
        card = card_index(card_char)
        next_index = start_index + 1
        if card_char == '1':
            next_index = start_index + 2
        suit_char = line[next_index]
        suit = suit_index(suit_char)
        index = card + NUM_CARDS*suit
        hand[index] = 1
        start_index = next_index + 2
    return player, hand

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

def extract(input_file):
    data = []
    
    f = open(input_file,"r")
    line = f.readline()
    line = f.readline()
    line.replace(" ", "")
    line = re.sub(r"[\n\t\s]*", "", line)
    secondChar = line[1]
    while secondChar != 'v':
        line = f.readline()
        line = re.sub(r"[\n\t\s]*", "", line)
        secondChar = line[1]
    vul_char = line[8]
    f.close()
    
    f = open(input_file,"r")
    f.readline()
    f.readline()
    line = f.readline()
    line = re.sub(r"[\n\t\s]*", "", line)
    dealer_char = line[10]
    dealer = player_index(dealer_char)
    f.readline()
    hands = np.zeros((NUM_PLAYERS,DECK_SIZE), dtype=int)
    player = -1
    for i in range(NUM_PLAYERS):
        line = f.readline()
        line = re.sub(r"[\n\t\s]*", "", line)
        next_player, hands[i,:] = hand_encoding(line)
        if next_player == dealer:
            player = i
    next_hand = hands[player,:]
    partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
    next_vul = vul_encoding(vul_char, player)
    bid_sequence = np.zeros((MAX_BID_LENGTH), dtype=int)
    f.readline()
    f.readline()
    line = f.readline()
    line = re.sub(r"[\n\t\s]*", "", line)
    bid_char = line[1]
    state = 0
    if bid_char == 'p':
        next_bid = 0
    else:
        tricks = int(line[1])
        suit = suit_index(line[2])
        next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
        bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
        state = -1
    data.append((next_hand, next_vul, np.copy(bid_sequence), next_bid, partner_hand))
    player = (player + 1)%NUM_PLAYERS
    next_hand = hands[player,:]
    partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
    next_vul = vul_encoding(vul_char, player)
    num_initial_passes = 0
    while state != -1:
        bid_sequence[num_initial_passes] = 1
        num_initial_passes = num_initial_passes + 1
        if num_initial_passes == 4:
            return data
        line = f.readline()
        line = re.sub(r"[\n\t\s]*", "", line)
        bid_char = line[1]
        if bid_char == 'p':
            next_bid = 0
        else:
            tricks = int(line[1])
            suit = suit_index(line[2])
            next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
            bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
            state = -1
        data.append((next_hand, next_vul, np.copy(bid_sequence), next_bid, partner_hand))
        player = (player + 1)%NUM_PLAYERS
        next_hand = hands[player,:]
        partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
        next_vul = vul_encoding(vul_char, player)
    while True:
        state = 0
        bid_sequence[bid_index] = 1
        line = f.readline()
        line = re.sub(r"[\n\t\s]*", "", line)
        bid_char = line[1]
        if bid_char == 'p':
            next_bid = 0
        elif bid_char == 'd':
            next_bid = 1
        elif bid_char == 'r':
            next_bid = 2
        else:
            tricks = int(line[1])
            suit = suit_index(line[2])
            next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
            bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
            state = -1
        data.append((next_hand, next_vul, np.copy(bid_sequence), next_bid, partner_hand))
        player = (player + 1)%NUM_PLAYERS
        next_hand = hands[player,:]
        partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
        next_vul = vul_encoding(vul_char, player)
        num_passes_1 = 0
        num_passes_2 = 0
        num_passes_3 = 0
        num_doubles = 0
        num_redoubles = 0
        while state != -1:
            if bid_char == 'd':
                bid_sequence[bid_index + 3] = 1
                num_doubles = 1
            elif bid_char == 'r':
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
            line = f.readline()
            line = re.sub(r"[\n\t\s]*", "", line)
            bid_char = line[1]
            if bid_char == 'p':
                next_bid = 0
            elif bid_char == 'd':
                next_bid = 1
            elif bid_char == 'r':
                next_bid = 2
            else:
                tricks = int(line[1])
                suit = suit_index(line[2])
                next_bid = 3 + (tricks-1)*(NUM_SUITS+1) + suit
                bid_index =  3 + ((tricks-1)*(NUM_SUITS+1) + suit)*9
                state = -1
            data.append((next_hand, next_vul, np.copy(bid_sequence), next_bid, partner_hand))
            player = (player + 1)%NUM_PLAYERS
            next_hand = hands[player,:]
            partner_hand = hands[(player + 2)%NUM_PLAYERS,:]
            next_vul = vul_encoding(vul_char, player)
    f.close()
    
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