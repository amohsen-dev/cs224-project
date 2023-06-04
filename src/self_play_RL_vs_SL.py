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
    path = play_random_game(agent1, agent2, verbose=True)
    print('IMP: ', path['rewards'][-1])
