# Playing puyopuyo with Reinforcement Learning
Tensorflow 2.0 implementation of playing puyopuyo with Reinforcement Learning. More details at [my blog(Korean)](https://queez0405.github.io/posts/)

## Algorithms
1. [ActorCritic](https://github.com/queez0405/puyopuyoRL/blob/master/ActorCriticpuyo.py)
2. [A2C](https://github.com/queez0405/puyopuyoRL/blob/master/A2Cpuyo.py)
3. [PPO](https://github.com/queez0405/puyopuyoRL/blob/master/new_PPopuyo.py): In progress

## Dependencies
Code runs on a single CPU and has been tested with
1. Python 3.5
2. Tensorflow 2.0
3. [gym_puyopuyo](https://github.com/frostburn/gym_puyopuyo)
4. Tensorboard
## Usage
```bash
# Works only with Python 3.
# e.g.
python3 ActorCriticpuyo.py
python3 A2Cpuyo.py
```

## Result

![Result](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/puyopuyo/puyo_result.gif)

## Training Detail

![Detail](https://raw.githubusercontent.com/queez0405/queez0405.github.io/master/_posts/puyopuyo/tensorboard_result.JPG)
Red: A2C
Blue: AC
