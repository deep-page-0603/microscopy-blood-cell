
from nn import NN
from env import Env
from common import *
import threading
import tensorflow as tf
import copy

""" Multithread Training Pipeline
    
    N_WORKER workers are running an agent with global net in global environment.
    If global game data counts more than BATCH_SIZE, pause all workers and starts update_thread.
    Every TEST_FREQ batches, evaluate global net by dueling with environment net.
    If advantage (= rate of (Victory Count - Lose Count)) exceeds UPDATE_RATIO, update environment by global net.
"""

N_WORKER = 8 # Number of workers
BATCH_SIZE = 500 # Size of game batches
TEST_SIZE = 500 # Size of test game batches
SAVE_FREQ = 1 # Save frequency
TEST_FREQ = 50 # Test frequency
LORD_ALTER_RATE = 0.3 # Alter ratio
FARMER_ALTER_RATE = 0.4 # Alter ratio

def update_thread():
    global GLOBAL_GAME_COUNTER, QUEUE, TEST_MODE
    global TRAIN_COUNTER, GLOBAL_SCORE, GLOBAL_NET, ENV_NET
    global WINS, LOSES, PREFIX, LORD_MODE

    while not COORD.should_stop():
        UPDATE_EVENT.wait() # Wait until batch is filled
        advantage = WINS - LOSES

        if TEST_MODE:
            print('\t(VIC : LOSE, ADVANTAGE : SCORE) = (' + str(WINS) +
                ' : ' + str(LOSES) + ', ' + str(advantage) + ' : ' + str(GLOBAL_SCORE) + ')\n')

            GLOBAL_NET.save('./model/global_' + PREFIX + '_net_(' + str(advantage) + ').model')

            alter_rate = LORD_ALTER_RATE if LORD_MODE else FARMER_ALTER_RATE
            
            if advantage >= TEST_SIZE * alter_rate:
                if LORD_MODE:
                    LORD_MODE = False
                    PREFIX = 'farmer'

                    GLOBAL_NET.save('./model/best_lord_net.model')
                    GLOBAL_NET.restore('./model/best_farmer_net.model')
                    ENV_NET.restore('./model/best_lord_net.model')

                    print('\n* Altering into Farmer Training Mode.\n')
                else:
                    LORD_MODE = True
                    PREFIX = 'lord'

                    GLOBAL_NET.save('./model/best_farmer_net.model')
                    GLOBAL_NET.restore('./model/best_lord_net.model')
                    ENV_NET.restore('./model/best_farmer_net.model')

                    print('\n* Altering into Lord Training Mode.\n')
            else:
                GLOBAL_NET.update(QUEUE)

            TEST_MODE = False
        else:            
            TRAIN_COUNTER += 1
            epoch, kl, lr_multiplier = GLOBAL_NET.update(QUEUE)

            print('* BATCH ' + str(TRAIN_COUNTER) + ' : GAMES = ' + str(len(QUEUE)) +
                '; (VIC : LOSE, ADVANTAGE : SCORE) = (' + str(WINS) +
                ' : ' + str(LOSES) + ', ' + str(advantage) + ' : ' + str(GLOBAL_SCORE) + '); ' +
                'EPOCH = ' + str(epoch) + ', KL = {:.3f}, LR_MULTIPLIER = {:.3f}'.format(kl, lr_multiplier))
            
            if TRAIN_COUNTER % SAVE_FREQ == 0:
                GLOBAL_NET.save('./model/global_' + PREFIX + '_net.model')
                #print('\n! ---> GLOBAL MODEL SAVED.\n')

            if TRAIN_COUNTER % TEST_FREQ == 0:
                print('\n\tEVALUATING ...')
                TEST_MODE = True

        QUEUE.clear()
        GLOBAL_SCORE = 0
        WINS = 0
        LOSES = 0
        
        GLOBAL_GAME_COUNTER = 0
        UPDATE_EVENT.clear()
        ROLLING_EVENT.set() # Reset roll event

""" Worker class """
class Worker(object):
    def __init__(self, wid):
        global GLOBAL_NET, ENV_NET, LORD_MODE

        self.net = GLOBAL_NET
        self.env = Env(ENV_NET, [0] if LORD_MODE else [1, 2]) # Position worker agent at a turn

    def work(self):
        global GLOBAL_GAME_COUNTER, GLOBAL_SCORE, TEST_MODE, QUEUE
        global WINS, LOSES, LORD_MODE

        while not COORD.should_stop():            
            self.env.agent = [0] if LORD_MODE else [1, 2]
            
            state, state_ex, done, reward = self.env.reset(not TEST_MODE) # Reset environment to launch new game
            buffer_state_ex = []
            
            while not done:
                if not ROLLING_EVENT.is_set(): # Wait until roll event is set
                    ROLLING_EVENT.wait()
                    buffer_state_ex = [] # If roll event is reset, clear buffers.
                    break

                act = self.net.choose_action(state_ex, not TEST_MODE, True) # Choose action from global net
                state_, state_ex_, done, reward = self.env.step(
                    self.env.board.legals[act], not TEST_MODE) # Step environment

                buffer_state_ex.append(state_ex[act])
                state, state_ex = state_, state_ex_
                
            if len(buffer_state_ex) == 0: continue

            #if not TEST_MODE:
            # Set reward sequence without discounting
            buffer_reward = (np.ones(len(buffer_state_ex)) * reward)[:, np.newaxis]
            QUEUE.append(np.hstack([
                np.vstack(buffer_state_ex),
                buffer_reward
            ]))

            GLOBAL_GAME_COUNTER += 1
            GLOBAL_SCORE += reward

            if reward > 0:
                WINS += 1
            else:
                LOSES += 1

            if GLOBAL_GAME_COUNTER >= (TEST_SIZE if TEST_MODE else BATCH_SIZE):
                # If global game counts more than batch size, reset update event.
                ROLLING_EVENT.clear()
                UPDATE_EVENT.set() # Set update event

if __name__ == '__main__':
    initialize_engine()
    print('\n\n* ', len(ALL_ACTIONS), ' Actions Loaded.')

    LORD_MODE = False

    if LORD_MODE:
        print('* Launched as Lord Training Mode.\n')

        PREFIX = 'lord'
        #GLOBAL_NET = NN('global')
        GLOBAL_NET = NN('global', './model/global_lord_net.model')
        #ENV_NET = NN('env')
        ENV_NET = NN('env', './model/best_farmer_net.model')
    else:
        print('* Launched as Farmer Training Mode.\n')

        PREFIX = 'farmer'
        #GLOBAL_NET = NN('global')
        GLOBAL_NET = NN('global', './model/global_farmer_net.model')
        #ENV_NET = NN('env')
        ENV_NET = NN('env', './model/best_lord_net.model')

    print('\n* Global and Environment Networks Loaded.\n')

    # Events
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()    
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    
    workers = [Worker(wid = i) for i in range(N_WORKER)]
    
    GLOBAL_GAME_COUNTER = 0
    GLOBAL_SCORE = 0
    WINS = 0
    LOSES = 0
    TEST_MODE = False
    TRAIN_COUNTER = 0

    COORD = tf.train.Coordinator()
    QUEUE = []
    threads = []

    # Start and join threads for workers and update
    for worker in workers:
        t = threading.Thread(target = worker.work, args = ())
        t.start()
        threads.append(t)

    threads.append(threading.Thread(target = update_thread, ))
    threads[-1].start()
    COORD.join(threads)