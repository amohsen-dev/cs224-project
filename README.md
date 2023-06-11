# cs224-project

# DATA
### Download Raw Data
run ```data/download.sh 1 100000``` to retrieve raw .lin files from vugraph

### Trim Raw Files
run ```data/trim.sh 1 100000``` to clean and trim the raw files

### LIN to JSON
run ```python data/linparse.py --start 1 --end 100000``` to parse trimmed lin files to human readable JSON 

### JSON to Feature Vectors
run ```python data/translation.py --start 1 --end 100000``` to extract vector features for learning from JSON into pandas dataframe

# Behavioral Cloning

### Train ENN
run ```python src/behavioral_cloning_ENN.py``` to pre-train the ENN model

### Train PNN
run ```python src/behavioral_cloning_PNN.py``` to pre-train the PNN model

# Reinforcement Learning

### Training
run the following to perform self-play for RL with different RL algorithms

```python src/self_play.py```
```python src/self_play.py --baseline```
```python src/self_play.py --ppo```
```python src/self_play.py --ppo --baseline```

### Evaluation vs BC
run the following to benchmark the RL agents vs the bevioral clone
```python src/self_play_RL_vs_SL.py```
```python src/self_play_PPO_AC_vs_SL.py```

# Play Against the bot
### Play Against bot on the Console
run the following for interactive play against the bot on the console
```python src/play_against_opponent.py```

### Play Against bot with our Bridge Bidding GUI
run the following to launch the GUI and play against the bot
```python src/gui/gui.py```
this will launch a window for interactive play. Note that the evaluation binary runs only on linux. Launching on MAC/Windows will work but will not output the final IMP score

