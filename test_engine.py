from Game_Engine.Wizard import Wizard
from Game_Engine.Player import RandomPlayer
import time


games = 5000

players = [RandomPlayer() for _ in range(4)]

initial = time.perf_counter()
last = initial

for i in range(games):
    if i % 100 == 0:
        print("{}/{}  time: {}".format(i, games, time.perf_counter() - last))
        last = time.perf_counter()
    wiz = Wizard(players=players)
    wiz.play()

print(time.perf_counter() - initial)