import os
import json
import argparse

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
    parser = argparse.ArgumentParser(description='parsing line files')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    args = parser.parse_args()
    for i in range(args.start, args.end + 1):
        with open(f"expert_data_final/games{i}_0.json", "r") as fp:
            dct = json.load(fp)
            clin = json_to_lin_cards(dct)
            eval = os.popen(f'../solver/bcalconsole -e e -q -t a -d lin -c {clin}').read()
            print(eval)