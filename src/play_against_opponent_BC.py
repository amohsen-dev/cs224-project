from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    agent1 = PNNAgent(stochastic=False,
                      pnn2=True,
                      path_pnn='../model_cache/model_pnn2/model_pnn_3.data')
    agent2 = ConsoleAgent()
    play_random_game(agent1, agent2, verbose=True)
