# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:56:37 2023

@author: samga
"""
import numpy as np
import argparse

North = 0
South = 2
East = 1
West = 3

Club = 0
Diamond = 1
Heart = 2
Spade = 3
Notrump = 4

duplicate_score_table = np.array([[70, 70, 140, 140, 230, 230],
    [80, 80, 160, 160, 520, 720],
    [90, 90, 180, 180, 560, 760],
    [90, 90, 180, 180, 560, 760],
    [110, 110, 470, 670, 640, 840],
    [120, 120, 490, 690, 680, 880],
    [110, 110, 470, 670, 640, 840],
    [140, 140, 530, 730, 760, 960],
    [400, 600, 550, 750, 800, 1000],
    [130, 130, 510, 710, 720, 920],
    [420, 620, 590, 790, 880, 1080],
    [430, 630, 610, 810, 920, 1120],
    [400, 600, 550, 750, 800, 1000],
    [450, 650, 650, 850, 1000, 1200],
    [460, 660, 670, 870, 1040, 1240],
    [920, 1370, 1090, 1540, 1380, 1830],
    [980, 1430, 1210, 1660, 1620, 2070],
    [990, 1440, 1230, 1680, 1660, 2110],
    [1440, 2140, 1630, 2330, 1960, 2660],
    [1510, 2210, 1770, 2470, 2240, 2940],
    [1520, 2220, 1790, 2490, 2280, 2980]])


# Change the duplicate scores to IMP
def calc_IMP(diff):
    if abs(diff) <= 10:
        imp = 0
    elif 20 <= abs(diff) <= 40:
        imp = 1
    elif 50 <= abs(diff) <= 80:
        imp = 2
    elif 90 <= abs(diff) <= 120:
        imp = 3
    elif 130 <= abs(diff) <= 160:
        imp = 4
    elif 170 <= abs(diff) <= 210:
        imp = 5
    elif 220 <= abs(diff) <= 260:
        imp = 6
    elif 270 <= abs(diff) <= 310:
        imp = 7
    elif 320 <= abs(diff) <= 360:
        imp = 8
    elif 370 <= abs(diff) <= 420:
        imp = 9
    elif 430 <= abs(diff) <= 490:
        imp = 10
    elif 500 <= abs(diff) <= 590:
        imp = 11
    elif 600 <= abs(diff) <= 740:
        imp = 12
    elif 750 <= abs(diff) <= 890:
        imp = 13
    elif 900 <= abs(diff) <= 1090:
        imp = 14
    elif 1100 <= abs(diff) <= 1290:
        imp = 15
    elif 1300 <= abs(diff) <= 1490:
        imp = 16
    elif 1500 <= abs(diff) <= 1740:
        imp = 17
    elif 1750 <= abs(diff) <= 1990:
        imp = 18
    elif 2000 <= abs(diff) <= 2240:
        imp = 19
    elif 2250 <= abs(diff) <= 2490:
        imp = 20
    elif 2500 <= abs(diff) <= 2990:
        imp = 21
    elif 3000 <= abs(diff) <= 3490:
        imp = 22
    elif 3500 <= abs(diff) <= 3990:
        imp = 23
    elif 4000 <= abs(diff):
        imp = 24
    else:
        imp = None
    return imp if diff >= 0 and imp is not None else -imp

# Calculate the duplicate score for the declarer's team
def calc_score(level, suit, trick, vulnerable, doubled, redoubled):
    if trick >= level + 6:
        over_trick = trick - level - 6
        if redoubled:
            if vulnerable == 1:
                score = duplicate_score_table[(level - 1) * 3 + int(suit) / 2][5] + over_trick * 400
            else:
                score = duplicate_score_table[(level - 1) * 3 + int(suit) / 2][4] + over_trick * 200
        elif doubled:
            if vulnerable == 1:
                score = duplicate_score_table[(level - 1) * 3 + int(suit) / 2][3] + over_trick * 200
            else:
                score = duplicate_score_table[(level - 1) * 3 + int(suit) / 2][2] + over_trick * 100
        else:
            score = duplicate_score_table[(level - 1) * 3 + int(suit) / 2][vulnerable]
            score += 30 * over_trick if suit >= Heart else 20 * over_trick
    else:
        under_trick = level + 6 - trick
        if doubled:
            if vulnerable == 1:
                score = 200 + 300 * (under_trick - 1)
            else:
                score = 100 + (200 * (under_trick - 1) if under_trick < 4 else 400 + 300 * (under_trick - 3))

            if redoubled:
                score *= 2
        else:
            score = 50 * under_trick * (vulnerable + 1)
    return score if trick >= level + 6 else -score


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Compute score')
    parser.add_argument('--pos1', type=int, default=-1,
                        help='pos1') #0 if we are N/S, 1 if we are E/W
    parser.add_argument('--level1', type=int, default=0,
                        help='level1')
    parser.add_argument('--suit1', type=int, default=-1,
                        help='suit1')
    parser.add_argument('--trick1', type=int, default=-1,
                        help='trick1') #From N/S perspective
    parser.add_argument('--declarer1', type=int, default=-1,
                        help='declarer1')
    parser.add_argument('--vul1', type=int, default=-1,
                        help='vul1')
    parser.add_argument('--doubled1', type=int, default=-1,
                        help='doubled1')
    parser.add_argument('--redoubled1', type=int, default=-1,
                        help='redoubled1')
    parser.add_argument('--pos2', type=int, default=-1,
                        help='pos2')
    parser.add_argument('--level2', type=int, default=0,
                        help='level2')
    parser.add_argument('--suit2', type=int, default=-1,
                        help='suit2')
    parser.add_argument('--trick2', type=int, default=-1,
                        help='trick2') #From N/S perspective
    parser.add_argument('--declarer2', type=int, default=-1,
                        help='declarer2')
    parser.add_argument('--vul2', type=int, default=-1,
                        help='vul2')
    parser.add_argument('--doubled2', type=int, default=-1,
                        help='doubled2')
    parser.add_argument('--redoubled2', type=int, default=-1,
                        help='redoubled2')
    args = parser.parse_args()
    
    pos1 = args.pos1
    level1 = args.level1
    suit1 = args.suit1
    trick1 = args.trick1
    declarer1 = args.declarer1
    vul1 = args.vul1
    doubled1 = args.doubled1
    redoubled1 = args.redoubled1
    
    pos2 = args.pos2
    level2 = args.level2
    suit2 = args.suit2
    trick2 = args.trick2
    declarer2 = args.declarer2
    vul2 = args.vul2
    doubled2 = args.doubled2
    redoubled2 = args.redoubled2
    
    score1 = calc_score(level1, suit1, trick1, vul1, doubled1, redoubled1)
    if declarer1 == East or declarer1 == West:
        score1 = -score1
    if (pos1 - declarer1)%2 != 0:
        score1 = -score1

    score2 = calc_score(level2, suit2, trick2, vul2, doubled2, redoubled2)
    if declarer2 == East or declarer2 == West:
        score2 = -score2
    if (pos2 - declarer2)%2 != 0:
        score2 = -score2
        
    result = calc_IMP(score1 + score2)   
    print(result)