import numpy as np
import pandas as pd
from tqdm import tqdm
from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    comparison = pd.DataFrame(columns=['dIMP', 'nSample', 'stdIMP'])
    for i in tqdm(range(0, 15360, 64)):
        try:
            agent1 = PNNAgent(
                path_enn=f'../model_cache3/RL/PG/model_enn_{i}.data',
                path_pnn=f'../model_cache3/RL/PG/model_pnn_{i}.data',
            )
            agent2 = PNNAgent(
                path_enn='../model_cache3/RL/PG/model_enn_0.data',
                path_pnn='../model_cache3/RL/PG/model_pnn_0.data',
            )

            imps = []
            for _ in range(128):
                try:
                    path = play_random_game(agent1, agent2, verbose=False)
                    imp = path['rewards'][-1]
                    imps.append(imp)
                except Exception:
                    pass
            comparison.loc[i] = (np.mean(imps), len(imps), np.std(imps))
            comparison.to_pickle('SL_vs_RL_comparison_3.pkl')
            print(comparison)
        except Exception:
            pass

