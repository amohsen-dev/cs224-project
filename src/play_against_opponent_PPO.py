from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    agent1 = PNNAgent(
        path_enn='../model_cache/RL/PPO/model_enn_1536.data',
        path_pnn='../model_cache/RL/PPO/model_pnn_1536.data',
    )
    agent2 = ConsoleAgent()
    play_random_game(agent1, agent2, verbose=True)
