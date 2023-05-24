import argparse
import pandas as pd
from extract_features_new import extract

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='translating json files')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    args = parser.parse_args()
    files = list(os.listdir('expert_data_final'))
    columns = [f'c{i}' for i in range(1, 53)] + ['v0', 'v1'] + [f'b{i}' for i in range(1, 319)] + ['bid']
    columns = columns + [f'cp{i}' for i in range(1, 53)]
    DFS = []
    for f in files[args.start:args.end]:
        fn += 1
        t_s = time.time()
        this_df = pd.DataFrame(columns=columns)
        if f.endswith('.json'):
            try:
                data = extract(f)  
                print(f'EVALUATED GAME # {fn}: time taken {time.time() - t_s:.3f}')
            except Exception as exception:
                print(exception)
