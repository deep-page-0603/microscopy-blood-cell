
import numpy as np
import random
from common import *

class Board():
    def __init__(self):
        self._init()

    def _init(self):
        self.legals = []
        self.hand = np.zeros((PCOUNT, 15))
        self.hidden = np.zeros(15, dtype = 'int32')
        self.turn = 0
        self.lt = 0
        self.la = PASS
        self.availables = [[], [], []]
        self.winner = []
        self.remaining = cards_to_hist(ALL_CARDS)
        self.deck = np.zeros((PCOUNT, 15))
        self.multiple = 1

    def reset(self):
        self._init()

        deck = ALL_CARDS.copy()
        random.shuffle(deck)

        for i in range(PCOUNT):
            hand = deck[:17]
            deck = deck[17:]
            self.hand[i][:] = cards_to_hist(hand)

        self.hidden = cards_to_hist(deck[:3])
        self._auto_start()

    def _get_hand_val(self):        
        hand_val = np.zeros(PCOUNT, dtype = 'float32')

        for i in range(PCOUNT):
            hand = list(self.hand[i])
            hand_val[i] = sum(hand) / 17 + len([card for card in set(hand) if hand.count(card) == 4]) * 3 \
                + len([card for card in set(hand) if hand.count(card) == 3]) + \
                (5 if 16 in hand and 17 in hand else 0)

        return hand_val

    def _start(self, lord):
        self.hand[lord] += self.hidden[:]
        self.hand = self.hand[[lord, (lord + 1) % PCOUNT, (lord + 2) % PCOUNT]]

        for i in range(PCOUNT):
            self.availables[i] = get_availables(self.hand[i])

        self.legals = self.availables[0]        
        return lord

    def _auto_start(self):
        #lord = np.argmax(self._get_hand_val())
        lord = random.randint(0, 2)
        return self._start(lord)

    """ Extract state info to feed value neural net """
    def extract_state(self):
        state = np.zeros(NN_S_DIM)
        offset = 0

        for i in range(PCOUNT):
            player = (self.turn + i) % PCOUNT

            if i == 0:
                state[offset:offset + 15] = self.hand[player][:]
            else:
                state[offset:offset + 15] = self.deck[player][:]

            if self.lt == player: state[offset + 15] = 1
            if player == 0: state[offset + 16] = 1

            offset += 17

        state[51:66] = ALL_ACTIONS[self.la][:15]
        state[66:81] = self.remaining[:] - self.hand[self.turn][:]
        state[81] = self.multiple
        return state

    """ Extract both state and extended state info to feed policy neural net """
    def extract(self):
        state = self.extract_state()
        state_ex = np.zeros((len(self.legals), NN_EX_DIM))
        
        for i in range(len(self.legals)):
            state_ex[i, :NN_S_DIM] = state[:]
            state_ex[i, NN_S_DIM:NN_S_DIM + 15] = ALL_ACTIONS[self.legals[i]][:15]
        
        return state, state_ex
    
    """ Progress board by action """
    def go(self, act):
        action = ALL_ACTIONS[act]

        if act != PASS:
            self.hand[self.turn] -= action[:15]
            self.remaining -= action[:15]
            self.deck[self.turn] += action[:15]

            if action[15] == AT_BOMB or action[15] == AT_BANG: self.multiple *= 2

            if np.sum(self.hand[self.turn]) == 0:
                self.winner = [0] if self.turn == 0 else [1, 2]

            self.availables[self.turn] = get_availables(self.hand[self.turn], self.availables[self.turn])
            self.lt = self.turn
            self.la = act

        self.turn = (self.turn + 1) % PCOUNT
        
        if self.turn == self.lt: self.la = PASS

        self.legals = get_legals(self.availables[self.turn], self.la)

    """ Check if game is over """
    def is_end(self):
        return self.winner != []

    """ Return winner list """
    def get_winner(self):
        return self.winner

    """ Return score list """
    def get_score(self, turn):
        return self.multiple if turn in self.winner else -self.multiple