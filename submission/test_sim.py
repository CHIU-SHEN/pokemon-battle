import os
import sys

sys.path.append(os.path.dirname(__file__))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
import random

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Provide a basic deck
        with open("deck.csv", "r") as f:
            return [int(line.strip()) for line in f if line.strip()]
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)

def main():
    deck0 = random_agent({'select': None, 'logs': [], 'current': None})
    deck1 = random_agent({'select': None, 'logs': [], 'current': None})
    
    obs, start_data = battle_start(deck0, deck1)
    
    print("Battle started! Error:", start_data.errorType)
    
    steps = 0
    while True:
        obs_obj = to_observation_class(obs)
        if obs_obj.current is not None and obs_obj.current.result != -1:
            print(f"Game over! Result: {obs_obj.current.result} after {steps} steps.")
            break
        
        # Which player is acting?
        player_idx = obs_obj.current.yourIndex
        
        # Call the appropriate agent
        if player_idx == 0:
            actions = random_agent(obs)
        else:
            actions = random_agent(obs)
            
        obs = battle_select(actions)
        steps += 1
        
        if steps > 1000:
            print("Too many steps!")
            break

    battle_finish()

if __name__ == "__main__":
    main()
