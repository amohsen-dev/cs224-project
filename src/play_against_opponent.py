from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    agent1 = PNNAgent(
        path_enn=f'../model_cache/RL/PG/model_enn_20480.data',
        path_pnn=f'../model_cache/RL/PG/model_pnn_20480.data',
        stochastic=False
    )
    agent2 = ConsoleAgent()
    play_random_game(agent1, agent2, verbose=True)
