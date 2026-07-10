import os
import sys

sys.path.append(os.path.dirname(__file__))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
import random

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        with open("deck.csv", "r") as f:
            return [int(line.strip()) for line in f if line.strip()]
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

def play_game(agent0, agent1):
    deck0 = agent0({'select': None, 'logs': [], 'current': None})
    deck1 = agent1({'select': None, 'logs': [], 'current': None})
    
    obs, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        battle_finish()
        raise ValueError(f"Battle failed to start. Error: {start_data.errorType}")
        
    steps = 0
    while True:
        obs_obj = to_observation_class(obs)
        if obs_obj.current is not None and obs_obj.current.result != -1:
            result = obs_obj.current.result
            break
        
        player_idx = obs_obj.current.yourIndex
        
        if player_idx == 0:
            actions = agent0(obs)
        else:
            actions = agent1(obs)
            
        obs = battle_select(actions)
        steps += 1
        
        if steps > 2000:
            result = 2 # Draw due to max steps
            break

    battle_finish()
    return result

def evaluate(agent0, agent1, num_games=100):
    wins0 = 0
    wins1 = 0
    draws = 0
    
    for _ in range(num_games):
        result = play_game(agent0, agent1)
        if result == 0:
            wins0 += 1
        elif result == 1:
            wins1 += 1
        else:
            draws += 1
            
    print(f"Agent 0 Wins: {wins0}")
    print(f"Agent 1 Wins: {wins1}")
    print(f"Draws: {draws}")
    return wins0, wins1, draws

if __name__ == "__main__":
    evaluate(random_agent, random_agent, num_games=10)
