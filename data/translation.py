import os
import time
import argparse
import numpy as np
import pandas as pd
from extract_features_new import extract

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog='Translate Data',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--start')
    parser.add_argument('-e', '--end')
    args = parser.parse_args()
    start, end = int(args.start), int(args.end)

    files = list(os.listdir('data/expert_data_final'))
    columns = [f'c{i}' for i in range(1, 53)] + ['v0', 'v1'] + [f'b{i}' for i in range(1, 319)] + ['bid']
    columns = columns + [f'cp{i}' for i in range(1, 53)]
    #eval_cols = [f'eval_{d}{i}' for d in ['N', 'S', 'E', 'W'] for i in range(5)]
    #columns = columns + eval_cols
    DFS = []
    for fn, f in enumerate(files[start:end]):
        t_s = time.time()
        this_df = pd.DataFrame(columns=columns, index=pd.MultiIndex(names=['GameID', 'BidID'], levels=[[],[]], codes=[[],[]]))
        if f.endswith('.json'):
            try:
                data = extract(os.path.join('data', 'expert_data_final', f))  
                for i, d in enumerate(data):
                    bid_data = np.concatenate(d)
                    #eval_data = np.zeros(20)
                    #if i == len(data) - 1:
                    #    with open(os.path.join('data', 'expert_data_eval', f.replace('json', 'txt')), 'r') as fp:
                    #        eval_data = fp.read().replace(' ', '').replace('\n', '').replace('N', '').replace('S', '').replace('E', '').replace('W', '')
                    #        eval_data = np.array([float(i) for i in eval_data])
                    #this_df.loc[(f.split('.')[0], i), :] = np.concatenate([bid_data, eval_data])
                    this_df.loc[(f.split('.')[0], i), :] = bid_data
                DFS.append(this_df)
            except Exception as exception:
                print(exception)
        print(f"translated file {fn}/{f}: took {time.time()-t_s:.2f} sec")
    pd.concat(DFS).to_pickle(f'DATA_{start}_{end}.pkl')
