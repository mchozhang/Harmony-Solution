# Harmony Solution
Deep Q-learning agent to solve the game puzzle [harmony](https://master.dg9tray1uvpm6.amplifyapp.com/),
using `tensorflow-agents`.

## Run the model
### Train a specify game level
```
# train level 30
python game/train.py 30

# train level 1 to 10
python game/train.py 1 10
```

### Evaluate the a policy of a level
```
# print the sequence of actions of level 11
python -m game 11
```

### Development Enviroment
```                            
# install tensorflow-agents, numpy and so forth
pip install -r requirements.txt
```

## Implementation
* Double Deep Q-learning agent
* Deep Q Network
* Uniform replay buffer
* Mask for legal actions

