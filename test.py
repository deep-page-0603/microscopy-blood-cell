
from env import Env
from nn import NN
from common import *

""" Simulate environment to test neural net """

def test_play():
	nn = NN('farmer', './model/best_farmer_net.model')
	#nn = NN('farmer')
	env = Env(nn, [0], is_render = True)
	lord_nn = NN('lord', './model/best_lord_net.model')
	#lord_nn = NN('lord')

	while True:
		_, state_ex, done, _ = env.reset(False)
		
		while not done:
			act = lord_nn.choose_action(state_ex, False)
			_, state_ex, done, _ = env.step(env.board.legals[act], False)

		x = input('')

if __name__ == '__main__':
	initialize_engine()
	test_play()