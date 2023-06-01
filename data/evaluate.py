import time
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
    fn = 0
    files = list(os.listdir('expert_data_final'))
    #for f in files[args.start:args.end]:
    for f in files[:1]:
        fn += 1
        t_s = time.time()
        if f.endswith('.json'):
            try:
                with open(f"expert_data_final/{f}", "r") as fp:
                    dct = json.load(fp)
                    clin = json_to_lin_cards(dct)
                    fw = f.replace('json', 'txt')
                    eval = os.popen(f'../solver/bcalconsole -e e -q -t a -d lin -c {clin} > expert_data_eval/{fw}').read()
                print(f'EVALUATED GAME # {fn}: time taken {time.time() - t_s:.3f}')
            except Exception as exception:
                print(exception)

