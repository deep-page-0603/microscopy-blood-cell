
from common import *
from nn import NN
from board import Board
import copy

""" Hilord Environment """

class Env():
	def __init__(self, nn, agent, is_render = False):
		self.nn = nn
		self.board = Board()
		self.agent = agent # Turn of agent
		self.is_render = is_render

	""" Reset environment """
	def reset(self, is_train):
		self.board.reset()

		if self.is_render:
			print('\n* Environment Reseted.\n')
			
			for i in range(PCOUNT):
				print(self._get_name(i), hist_to_str(self.board.hand[i]))

			print('\n')

		return self._progress(is_train)

	def _render_result(self):
		if not self.is_render: return
		if self.board.get_winner() == [0]:
			print('\n! Lord Won. Score = ' + str(self.board.multiple) + '\n')
		else:
			print('\n! Farmers Won. Score = ' + str(self.board.multiple) + '\n')

	""" Progress game until agent's turn or game over """
	def _progress(self, is_train):
		end = False

		while self._go(is_train) > -1:
			end = self.board.is_end()
			if end: break

		if end:
			self._render_result()
			return None, None, True, self.board.get_score(self.agent[0])			
		else:
			state, state_ex = self.board.extract()
			return state, state_ex, False, 0

	""" Step game using environment net """
	def _go(self, is_train):
		if len(self.board.legals) == 1:
			act = self.board.legals[0]
		elif self.board.turn not in self.agent:
			_, state_ex = self.board.extract()
			act = self.board.legals[self.nn.choose_action(state_ex, is_train)]
		else:
			return -1

		if self.is_render:
			action = 'PASS' if act == PASS else hist_to_str(ALL_ACTIONS[act])
			print(self._get_name(self.board.turn),
				hist_to_str(self.board.hand[self.board.turn] - ALL_ACTIONS[act][:15]) + ' : ' + action)

		self.board.go(act)
		return act

	def _get_name(self, turn):
		return '[L]' if turn == 0 else '[F' + str(turn) + ']'

	""" Step environment """
	def step(self, act, is_train):
		if self.is_render:
			action = 'PASS' if act == PASS else hist_to_str(ALL_ACTIONS[act])

			print(self._get_name(self.board.turn),
				hist_to_str(self.board.hand[self.board.turn] - ALL_ACTIONS[act][:15]) + ' : ' + action)

		self.board.go(act)
		
		if self.board.is_end():
			self._render_result()
			return None, None, True, self.board.get_score(self.agent[0])
		else:
			return self._progress(is_train)