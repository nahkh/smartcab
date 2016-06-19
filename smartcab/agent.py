import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from sets import Set


class State:
    """A representation of the current state of the agent"""
    
    
    def __init__(self):
        self.valid_lights = ['red', 'green']
        self.valid_traffic_states = [None, 'left', 'right', 'forward']
        # We omit the None direction, as we'll have arrived and the simulation for that round ends
        self.valid_directions = ['left', 'right', 'forward']
        self.light = 'red'
        self.traffic_left = None
        self.traffic_right = None
        self.traffic_oncoming = None
        self.desired_direction = None
        
        # Total number of possible states is 2 * 4 * 4 * 4 * 3 = 384

        
    def set_lights(self, light):
        self.light = light
    
    def set_oncoming_traffic(self, oncoming):
        self.traffic_oncoming = oncoming
        
    def set_left_traffic(self, left):
        self.traffic_left = left
    
    def set_right_traffic(self, right):
        self.traffic_right = right
        
    def set_desired_direction(self, direction):
        self.desired_direction = direction
        
    def update_from_input(self, next_waypoint, inputs):
        self.set_lights(inputs['light'])
        self.set_left_traffic(inputs['left'])
        self.set_right_traffic(inputs['right'])
        self.set_oncoming_traffic(inputs['oncoming'])
        self.set_desired_direction(next_waypoint)
    
    # Calculate the id of the state
    def get_state_id(self):
        id = self.valid_lights.index(self.light)
        id += 2 * self.valid_traffic_states.index(self.traffic_right)
        id += 8 * self.valid_traffic_states.index(self.traffic_left)
        id += 32 * self.valid_traffic_states.index(self.traffic_oncoming)
        id += 128 * self.valid_directions.index(self.desired_direction)
        return id
        
def run_state_test():
    state = State()
    seen_ids = Set()
    for light in state.valid_lights:
        state.set_lights(light)
        for left in state.valid_traffic_states:
            state.set_left_traffic(left)
            for right in state.valid_traffic_states:
                state.set_right_traffic(right)
                for oncoming in state.valid_traffic_states:
                    state.set_oncoming_traffic(oncoming)
                    for direction in state.valid_directions:
                        state.set_desired_direction(direction)
                        seen_ids.add(state.get_state_id())
    for i in range(384):
        if i not in seen_ids: raise AssertionError(i)

    

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env, learning_rate, discount_rate):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.state = State()
        self.q_hat = {}
        self.cumulative_reward = 0
        self.total_reward = 0
        self.runs = 0
        self.random_chance = 0.05
        self.random_chance_decay = 0.99
        self.total_steps = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.total_reward += self.cumulative_reward
        if self.runs > 0:
            self.random_chance *= self.random_chance_decay
            print "Resetting agent. Current cumulative reward: %.2f, Achieved average reward %.2f" % (self.cumulative_reward, self.total_reward / self.runs)
        self.runs += 1
        self.cumulative_reward = 0
        
    def calculate_utility(self, state):
        state_id = state.get_state_id()
        if state_id in self.q_hat:
            max = 0;
            for k, v in self.q_hat[state_id].items():
                if v > max:
                    max = v
            return max          
        else:
            return 0
    
    def act_randomly(self):
        return random.random() < self.random_chance

    def find_best_action(self):
        # Sometimes we need to act randomly to make sure we don't get stuck in local optima
        if self.act_randomly():
            return random.choice(self.env.valid_actions)
    
        # We find the best action to take if we've tried each action at least once in this state
        # If we run into a situation where we haven't tried everything we return one of the untried ones instead
        state_id = self.state.get_state_id();
        if state_id in self.q_hat:
            known_q_value = self.q_hat[state_id]
        else:
            known_q_value = {}
        
        if 'left' not in known_q_value:
            return 'left'
        if 'right' not in known_q_value:
            return 'right'
        if 'forward' not in known_q_value:
            return 'forward'
        if None not in known_q_value:
            return None
        
        max = 0
        max_action = None
        for k, v in known_q_value.items():
            if(v > max):
                max = v
                max_action = k
        return max_action
        
    def update_q_hat(self, action, reward, utility_of_new_state):
        state_id = self.state.get_state_id()
        if state_id in self.q_hat:
            known_q_value = self.q_hat[state_id]
        else:
            known_q_value = {}
        
        if action in known_q_value:
            known_q_value_for_action = known_q_value[action]
        else: 
            known_q_value_for_action = 0
        
        known_q_value[action] = (1.0 - self.learning_rate) * known_q_value_for_action + self.learning_rate * (reward + self.discount_rate * utility_of_new_state)
        self.q_hat[state_id] = known_q_value
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state.update_from_input(self.next_waypoint, inputs)
        
        # Select action according to your policy
        action = self.find_best_action() #random.choice(self.env.valid_actions)
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        
        # Learn policy based on state, action, reward
        
        # Evaluate new state
        # TODO refactor the update method so we could reuse this newly calculated state in the next iteration
        new_next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        new_state = State()
        new_state.update_from_input(new_next_waypoint, inputs)
        
        new_state_utility = self.calculate_utility(new_state)
        # Learn 
        self.update_q_hat(action, reward, new_state_utility)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.total_steps += 1
        
    def get_score(self):
        return self.total_reward / float(self.total_steps)
        
def run_grid_search():
    """Run a gridsearch to determine optimal parameters for the agent"""

    best_score = 0
    best_learning_rate = 0
    best_discount_rate = 0
    
    # TODO These ought to be done with numpy.arange but I don't have that package installed at the moment
    for learning_rate_raw in range(1, 50, 1):
        for discount_rate_raw in range(1, 20, 1):
            learning_rate = learning_rate_raw * 0.01
            discount_rate = discount_rate_raw * 0.05
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent, learning_rate, discount_rate)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=30)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            score = a.get_score()
            if score > best_score:
                best_score = score
                best_learning_rate = learning_rate
                best_discount_rate = discount_rate
                
    print "Gridsearch finished, best learning rate: %.2f, best discount rate: %.2f" % (best_learning_rate, best_discount_rate)
    
def run():
    """Run the agent for a finite number of trials."""  
    learning_rate = 0.42
    discount_rate = 0.15
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, learning_rate, discount_rate)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


    
if __name__ == '__main__':
    run()
