# Applying Reinforcement Learning to the Card Game Wizard

This repo contains the implementation for an agent learning to play the card game [wizard](https://en.wikipedia.org/wiki/Wizard_(card_game)) through Reinforcement Learning. In the current implementation a version of Proximal Policy Optimization was used. However, it is trivial to use other algorithms for learning with the current setup.

This work was created during the course _Neural Information Processing Projects_ at TU Berlin in the spring term 2019. Supervisor was Vaios Laschos

## Setup

Install dependencies: \
```pip install -r requirements.txt```

## Repo Structure

* agents --> contains the implementation for the rl agents and rule based agent. tf-agents library is used
	* contains implementation for predictors, featurizers  	
* evaluation --> saves models and logs during evaluation process
* game engine --> contains implementation of game logic 
* tests --> contains code for training and evaluation the agent
* wizard_site --> contains implementation for GUI. Used for evaluating agents against human players

## Usage

To train the agent: \   
```$ python3 tests/test_train_and_evaluate.py train_vs_old_self```

To evaluate the agent: \  
```$ python3 tests/test_train_and_evaluate.py evaluate```

The above commands will also start a tensorboard session on port 6006.  

Also see the comments and docstrings in _tests/test_train_and_evaluate.py_


To start the GUI: \  
From the root directory run: ```python wizard_site/manage.py runserver```

## Paper

For more detailed information on the implementation and results check the paper and poster in _[/paper](/paper)_

## Contributers

Jonas Dippel, Callum Waters, Til Jasper Ullrich, Kai Jeggle