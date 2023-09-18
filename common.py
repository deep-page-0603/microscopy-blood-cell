
import numpy as np
import random
import itertools as it

""" Hitlord Common Module contains the followings:
	global constants
	functions to initialize all Hitlord available actions
	implementation of Hitlord game logic
	functions to export trained neural net
"""

CARD = {'3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10,
	'J' : 11, 'Q' : 12, 'K' : 13, 'A' : 14, '2' : 15, '-' : 16, '+' : 17}

FACE = {3 : '3', 4 : '4', 5 : '5', 6 : '6', 7 : '7', 8 : '8', 9 : '9', 10 : '10',
	11 : 'J', 12 : 'Q', 13 : 'K', 14 : 'A', 15 : '2', 16 : '-', 17 : '+'}

# Action types
AT_PASS = 0; AT_SINGLE = 1; AT_PAIR = 2; AT_THREE = 3; AT_THREE_SINGLE = 4; AT_THREE_PAIR = 5;
AT_SEQ = 6; AT_DOUBLE_SEQ = 7; AT_TRIPLE_SEQ = 8; AT_TRIPLE_SEQ_SINGLE = 9; AT_TRIPLE_SEQ_PAIR = 10;
AT_BOMB = 11; AT_BOMB_SINGLE = 12; AT_BOMB_PAIR = 13; AT_BANG = 14;

# Game setting
PCOUNT = 3;

# Action globals
ALL_ACTIONS = []
ACTION_TYPE_RANGE = []
ACTION_TYPE_LEN = [0, 1, 2, 3, 4, 5, -5, -6, -6, -8, -10, 4, 5, 6, 2]
ALL_CARDS = sorted(list(range(3, 16)) * 4 + [16, 17])
ACTION_COUNT = 12655

# Special actions
PASS = 0
BANG = ACTION_COUNT - 1

# Dimensions of neural net
NN_S_DIM = 82
NN_EX_DIM = NN_S_DIM + 15
NN_POLICY_DEPTH = 10

# Hyper parameters
NN_C_LR = 0.00001 # Critic learning rate
NN_UPDATE_STEP = 5 # Update loops
NN_TARG = 0.03 # Clipping boundary

ARGMAX_MODE = True

""" Convert histogram to action id """
def hist_to_act(hist):
	if len(hist) > 15:
		ats = [hist[15]]
	else:
		actlen = np.sum(hist[:15])
		ats = []
		for i in range(15):
			if actlen == ACTION_TYPE_LEN[i]:
				ats.append(i)
			elif ACTION_TYPE_LEN[i] < 0 and actlen >= (-ACTION_TYPE_LEN[i]):
				ats.append(i)
	for at in ats:
		for act in ACTION_TYPE_RANGE[at]:
			if np.min(hist[:15] == ALL_ACTIONS[act][:15]) == True:
				return act
	return None

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def hist_to_cards(hist):
	cards = []
	for i in range(15):
		cards.extend([i + 3] * int(hist[i]))
	return cards

def cards_to_hist(cards, action_type = None):
	hist = np.zeros(15, dtype = 'int32')
	for i in range(15):
		hist[i] = cards.count(i + 3)
	if action_type:
		hist = np.hstack([hist, action_type])
	return hist

def cards_to_faces(cards):
	return [FACE[a] for a in cards];

def faces_to_cards(faces):
	return [CARD[a] for a in faces];

def hist_to_faces(hist):
	return cards_to_faces(hist_to_cards(hist))

def hist_to_str(hist):
	return ' '.join(cards_to_faces(hist_to_cards(hist)))

def cards_to_str(cards):
	return ' '.join(cards_to_faces(sorted(cards)))

def faces_to_hist(faces):
	return cards_to_hist(faces_to_cards(faces))

def faces_to_act(faces):
	return hist_to_act(faces_to_hist(faces))

def initialize_engine():
	ALL_ACTIONS.clear()
	ACTION_TYPE_RANGE.clear()
	start = 0
	
	action_funcs = [get_passes, get_singles, get_pairs, get_threes, get_three_singles, get_three_pairs, get_seqs,
		get_double_seqs, get_triple_seqs, get_triple_seq_singles, get_triple_seq_pairs, get_bombs,
		get_bomb_singles, get_bomb_pairs, get_bangs]

	for i in range(len(action_funcs)):
		ALL_ACTIONS.extend(action_funcs[i]())
		ACTION_TYPE_RANGE.append(range(start, len(ALL_ACTIONS)))
		start = len(ALL_ACTIONS)

def get_passes():
	return [np.hstack([np.zeros(15, dtype = 'int32'), AT_PASS])]

def get_repeat_actions(repeat, act_type):
	result = []
	for i in range(15):
		if repeat > 1 and i > 12: break
		action = np.zeros(16, dtype = 'int32')
		action[i] = repeat
		action[15] = act_type
		result.append(action)
	return result

def get_singles():
	return get_repeat_actions(1, AT_SINGLE)

def get_pairs():
	return get_repeat_actions(2, AT_PAIR)

def get_threes():
	return get_repeat_actions(3, AT_THREE)

def get_bombs():
	return get_repeat_actions(4, AT_BOMB)

def get_bangs():
	return [np.hstack([np.zeros(13, dtype = 'int32'), np.ones(2, dtype = 'int32'), AT_BANG])]

def get_three_singles():
	result = []
	for i in range(13):
		for j in range(15):
			if i == j: continue
			action = np.zeros(16, dtype = 'int32')
			action[i] = 3
			action[j] = 1
			action[15] = AT_THREE_SINGLE
			result.append(action)
	return result

def get_bomb_singles():
	result = []
	for i in range(13):
		for j in range(14):
			for k in range(j + 1, 15):
				if i == j or i == k: continue
				if j == 13 and k == 14: continue

				action = np.zeros(16, dtype = 'int32')
				action[i] = 4
				action[j] = action[k] = 1
				action[15] = AT_BOMB_SINGLE
				result.append(action)
	return result

def get_bomb_pairs():
	result = []
	for i in range(13):
		for j in range(12):
			for k in range(j + 1, 13):
				if i == j or i == k: continue
				action = np.zeros(16, dtype = 'int32')
				action[i] = 4
				action[j] = action[k] = 2
				action[15] = AT_BOMB_PAIR
				result.append(action)
	return result

def get_three_pairs():
	result = []
	for i in range(13):
		for j in range(13):
			if i == j: continue
			action = np.zeros(16, dtype = 'int32')
			action[i] = 3
			action[j] = 2
			action[15] = AT_THREE_PAIR
			result.append(action)
	return result

def get_seqs():
	result = []
	for i in range(5, 13):
		for j in range(13 - i):
			action = np.zeros(16, dtype = 'int32')
			action[j:j + i] = 1
			action[15] = AT_SEQ
			result.append(action)
	return result

def get_double_seqs():
	result = []
	for i in range(3, 11):
		for j in range(13 - i):
			action = np.zeros(16, dtype = 'int32')
			action[j:j + i] = 2
			action[15] = AT_DOUBLE_SEQ
			result.append(action)
	return result

def get_triple_seqs():
	result = []
	for i in range(2, 7):
		for j in range(13 - i):
			action = np.zeros(16, dtype = 'int32')
			action[j:j + i] = 3
			action[15] = AT_TRIPLE_SEQ
			result.append(action)
	return result

def get_triple_seq_singles():
	result = []
	for i in range(2, 6):
		for j in range(13 - i):
			arm_set = set(range(15)) - set(range(j, j + i))
			arms = list(it.combinations(arm_set, i))
			for arm in arms:
				if 13 in arm and 14 in arm: continue
				action = np.zeros(16, dtype = 'int32')
				action[j:j + i] = 3
				action[list(arm)] = 1
				action[15] = AT_TRIPLE_SEQ_SINGLE
				result.append(action)
	return result

def get_triple_seq_pairs():
	result = []
	for i in range(2, 5):
		for j in range(13 - i):
			arm_set = set(range(13)) - set(range(j, j + i))
			arms = list(it.combinations(arm_set, i))
			for arm in arms:
				action = np.zeros(16, dtype = 'int32')
				action[j:j + i] = 3
				action[list(arm)] = 2
				action[15] = AT_TRIPLE_SEQ_PAIR
				result.append(action)
	return result

def is_available(act, hand):
	action = ALL_ACTIONS[act]

	for i in range(15):
		if hand[i] < action[i]:	return False

	return True

def get_availables(hand, actions = None):
	if actions is None:
		actions = list(range(1, ACTION_COUNT))

	return [action for action in actions if is_available(action, hand)]

def get_legals(availables, la = PASS):
	if la == PASS: return availables
	
	last_action = ALL_ACTIONS[la]
	last_len = np.sum(last_action[:15])
	legals, superiors = [], []

	if la != BANG:
		superiors.append(AT_BANG)

		#if last_action[15] not in [AT_BOMB, AT_BOMB_SINGLE, AT_BOMB_PAIR]:
		if last_action[15] != AT_BOMB:
			superiors.append(AT_BOMB)	

	for act in availables:
		action = ALL_ACTIONS[act]

		if action[15] in superiors:	legals.append(act)
		elif action[15] == last_action[15] and act > la and (np.sum(action[:15]) == last_len):
			legals.append(act)

	return legals + [PASS]

""" Produce C++ file containing all actions """
def port_all_actions_to_cpp():
	f = open('d:/all_actions.cpp', 'w')
	f.write('{')

	for i in range(ACTION_COUNT):
		f.write('{')

		for j in range(15):
			f.write(str(ALL_ACTIONS[i][j]) + ', ')

		f.write(str(ALL_ACTIONS[i][15]) + '},\n')

	f.write('}')
	f.close()

""" Produce C++ file containing policy neural net variables and matrix shapes """
def port_net_to_cpp(network, prefix, scope, depth, filename):
    variables = network.get_variables(scope, depth)
    column = 16
    var_count = 0

    for i in range(depth):    	
    	var_count += variables[i][0].shape[0] * variables[i][0].shape[1] + variables[i][1].shape[0]

    f = open(filename, 'w');

    f.write('\n#include \"common.h\"\n\n')
    f.write('int _' + prefix + '_net_shapes[' + str(depth * 2) + '] = {')

    for i in range(depth):
    	f.write(str(variables[i][0].shape[0]) + ', ')
    	f.write(str(variables[i][0].shape[1]) + (', ' if i != depth - 1 else ''))
    
    f.write('};\n')
    f.write('float _' + prefix + '_net_vars[' + str(var_count) + '] = {')

    var_count = 0

    for i in range(depth):
    	# Writing matrix shapes
    	for j in range(variables[i][0].shape[0]):
    		for k in range(variables[i][0].shape[1]):
    			f.write(str(variables[i][0][j][k]) + 'f, ')
    			var_count += 1
    			if (var_count % column == 0):
    				f.write('\n')
    	# Writing variables
    	for j in range(variables[i][1].shape[0]):
    		f.write(str(variables[i][1][j]) +
    			('f, ' if i != depth - 1 or j != variables[i][1].shape[0] - 1 else 'f'))
    		var_count += 1
    		if (var_count % column == 0):
    			f.write('\n')
    
    f.write('};')
    f.close()

    return var_count

def port_net_to_txt(network, net_scope, depth, filename):
	variables = network.get_variables(net_scope, depth)
	var_count = 0

	for i in range(depth):
		var_count += variables[i][0].shape[0] * variables[i][0].shape[1] + variables[i][1].shape[0]

	f = open(filename, 'w');

	f.write(str(depth) + '\n')
	f.write(str(var_count) + '\n')

	for i in range(depth):
		f.write(str(variables[i][0].shape[0]) + '\n')
		f.write(str(variables[i][0].shape[1]) + '\n')

	for i in range(depth):
		for j in range(variables[i][0].shape[0]):
			for k in range(variables[i][0].shape[1]):
				f.write(str(variables[i][0][j][k]) + '\n')

		for j in range(variables[i][1].shape[0]):
			f.write(str(variables[i][1][j]) + '\n')

	f.close()