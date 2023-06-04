import pandas as pd
from tqdm import tqdm
from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    agent1 = PNNAgent(
        path_enn='../model_cache/RL/PG/model_enn_31232.data',
        path_pnn='../model_cache/RL/PG/model_pnn_31232.data',
    )
    agent2 = PNNAgent(
        path_enn='../model_cache/RL/PG/model_enn_0.data',
        path_pnn='../model_cache/RL/PG/model_pnn_0.data',
    )

    imps = []
    for i in tqdm(range(64)):
        path = play_random_game(agent1, agent2, verbose=False)
        imp = path['rewards'][-1]
        imps.append(imp)

    print(pd.Series(imps).to_frame('IMP').expanding().std())
    print(pd.Series(imps).to_frame('IMP').expanding().mean())

