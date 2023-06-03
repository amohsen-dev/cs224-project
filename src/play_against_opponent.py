from self_play import ConsoleAgent, PNNAgent, play_random_game

if __name__ == '__main__':
    agent1 = PNNAgent()
    agent2 = ConsoleAgent()
    play_random_game(agent1, agent2)
