import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString

'''
Example City class
class City:
    def __init__(self, name, geolocation, other_features=None):
        self.name = name
        self.geolocation = geolocation  # A tuple (latitude, longitude)
        self.other_features = other_features if other_features is not None else {}
'''
'''
Example model usage
agent = QLearningAgent(...)  (Initialize the agent)
agent.setup_environment() (Initialise the Q-table)
agent.train()  (Train the agent)
optimal_policy = agent.get_optimal_policy()  (Get the optimal policy)
total_reward = agent.evaluate_policy(optimal_policy)  (Evaluate the optimal policy)
print("Total Reward:", total_reward)
agent.visualize_route(optimal_policy)  (Visualize the optimal route)
'''


class QLearningAgent:

####################################################################################################################
#                                        Init and accessory methods                                                #
####################################################################################################################

    def __init__(self, learning_rate, discount_factor, exploration_rate, current_city, num_episodes, cities,
                 initial_budget, total_time, user_preferences, desired_cities, transportation_scores):
        """
        Initialize the Q-Learning Agent.

        learning_rate: The rate at which the Q-table is updated.
        discount_factor: The discount factor for future rewards.
        exploration_rate: The rate at which the agent chooses a random action for exploration.
        num_episodes: The number of episodes for the agent to learn from.
        cities: A list of City objects.
        initial_budget: The initial budget for the trip.
        initial_total_time: Store the initial total time
        total_time: The total time for the trip.
        user_preferences: The user's preferences for each city.
        desired_cities: The cities the user wants to visit. If None, the user has no preference.
        transportation_scores: A dictionary that indicates the "convenience factor" for each transportation method
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.current_city = current_city
        self.num_episodes = num_episodes
        self.cities = cities
        self.num_cities = len(cities)
        self.initial_budget = initial_budget
        self.budget = initial_budget
        self.initial_total_time = total_time
        self.total_time = total_time
        self.user_preferences = user_preferences
        self.desired_cities = desired_cities if desired_cities is not None else list(range(self.num_cities))
        self.transportation_scores = transportation_scores
        '''
        Example of priorly defined transportation_score dictionary (could be country-based)
        transportation_scores = {
            'airplane': {'cost_per_km': 0.1, 'time_per_km': 0.01},
            'train': {'cost_per_km': 0.05, 'time_per_km': 0.02},
            'car': {'cost_per_km': 0.07, 'time_per_km': 0.03},
            ...
        }
        '''
        self.q_table = None
        self.penalty = None
        self.model_prior_features = None
        self.cost_per_km = None
        self.time_per_km = None

    def setup_environment(self):
        # Initialize the Q-table with zeros.
        self.q_table = np.zeros((self.num_cities * 10 * 10, self.num_cities))

    def update_exploration_rate(self, episode):
        """
        Update the exploration rate based on the current episode.
        This function decreases the exploration rate over time.
        """
        # Linear decay
        decay_rate = 0.01
        self.exploration_rate -= decay_rate

        # Optionally, exponential decay can be used
        # decay_factor = 0.99
        # self.exploration_rate *= decay_factor

        # Ensure that the exploration rate does not go below a minimum threshold
        min_exploration_rate = 0.1
        self.exploration_rate = max(self.exploration_rate, min_exploration_rate)

    def calculate_distance(self, city1, city2):
        """
        Calculate the distance between two cities.

        city1: The first city.
        city2: The second city.

        This function returns the distance between the two cities.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [city1.geolocation[0], city1.geolocation[1], city2.geolocation[0], city2.geolocation[1]])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

####################################################################################################################
# #                                          Model's methods                                                       #
####################################################################################################################

    def get_state(self):
        # One-hot encode the current city.
        city_state = self.current_city

        # Discretize the budget and total time into intervals.
        budget_interval = self.initial_budget // 10  # Divide the budget into 10 intervals.
        time_interval = self.total_time // 10  # Divide the total time into 10 intervals.

        budget_state = int(self.budget // budget_interval)
        time_state = int(self.total_time // time_interval)

        # Combine the city, budget, and time states into a single state.
        state = city_state * 100 + budget_state * 10 + time_state

        return state

    def get_possible_actions(self):
        """
        Get the possible actions that can be taken from the current state.
        This function returns a list of possible actions.
        """
        # Return a list of possible actions. In this case, the actions are specific city moves.
        # If the user has specified desired cities, only return actions that lead to these cities.
        return [action for action in range(self.num_cities) if action in self.desired_cities]

    def choose_action(self, state):
        """
        Choose an action based on the current state and the Q-table.
        This function returns the chosen action.
        """
        # If a randomly chosen number is less than the exploration rate, choose a random action.
        if np.random.rand() < self.exploration_rate:
            action = np.random.choice(self.get_possible_actions())
        else:
            # Otherwise, choose the action with the highest Q-value for the current state.
            action = np.argmax(self.q_table[state])
        return action

    def calculate_reward(self, current_city_index, new_city_index, user_preferences, model_prior_features, alpha, beta):
        # Choose the mode of transportation with the highest score.
        transportation_mode = max(self.transportation_scores, key=self.transportation_scores.get)

        # Distance between current city and new city
        distance = self.calculate_distance(self.cities[current_city_index], self.cities[new_city_index])

        # Calculates cost and time based on given parameters. Those parameters vary on the type of transportation between the cities
        cost = distance * self.transportation_scores[transportation_mode]['cost_per_km']
        time = distance * self.transportation_scores[transportation_mode]['time_per_km']

        # Check if the move is possible with the current budget and time.
        if cost > self.budget or time > self.total_time:
            return -np.inf  # Return a large negative reward if the move is not possible.

        # Daily reward for staying in the current city
        daily_reward = user_preferences[current_city_index] + model_prior_features[current_city_index]

        # Cost for moving to a new city
        moving_cost = -distance

        # Total reward for moving to a new city or staying in the current one
        if current_city_index == new_city_index:
            reward = daily_reward  # Staying in the current city
        else:
            reward = daily_reward + moving_cost  # Moving to a new city

        # Normalize the reward by the total time of the trip
        reward /= self.total_time

        # Apply a discount factor to the reward based on the distance between the current city and the new city.
        reward *= self.discount_factor ** distance

        # Update the budget and time.
        self.budget -= cost
        self.total_time -= time

        # Give a large positive reward if the budget is optimally used.
        if self.budget <= beta * self.initial_budget:  # Adjust this threshold as needed.
            reward += alpha  # Adjust this reward as needed.

        return reward

    def move_to_new_city(self, action):
        """
        Move to a new city based on the chosen action.
        This function returns the new state and reward.
        """
        # The new city index is the chosen action.
        new_city_index = action

        # Calculate the reward for moving to the new city.
        reward = self.calculate_reward(self.current_city, new_city_index, self.user_preferences,
                                       self.model_prior_features)

        # Update the current city to the new city.
        self.current_city = new_city_index

        # The new state is the state after moving to the new city.
        new_state = self.get_state()

        return new_state, reward

    def stay_in_current_city(self):
        """
        Stay in the current city.
        This function returns the new state and reward.
        """
        # The new city is the current city.
        new_city = self.current_city

        # Calculate the reward for staying in the current city.
        # This could be a fixed amount or it could be based on some factors (e.g., the user's preferences or the city's features).
        reward = self.calculate_reward(self.current_city, new_city, self.user_preferences, self.model_prior_features)

        # The new state is the state after staying in the current city.
        new_state = self.get_state()

        return new_state, reward

    def take_action(self, action, state):
        """
        Take the chosen action and return the new state and reward.

        action: The action to take (move to a specific city).
        state: The current state.

        This function should return a tuple (new_state, reward).
        """
        if action != self.current_city:  # Move to a new city
            return self.move_to_new_city(action)
        else:  # Stay in the current city
            return self.stay_in_current_city()

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the state, action, reward, and new state.

        state: The current state.
        action: The action taken.
        reward: The reward received.
        new_state: The state after taking the action.
        """
        # Implement the Q-learning update rule.
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (
                                              reward + self.discount_factor * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the Q-Learning agent.
        This function implements the main Q-Learning algorithm.
        """
        # Implement the main Q-Learning algorithm.
        for episode in range(self.num_episodes):
            # Reset the state, budget, and time at the start of each episode
            self.current_city = np.random.choice(self.num_cities)  # Start at a random city
            self.budget = self.initial_budget
            self.total_time = self.initial_total_time  # Reset total time to the initial value

            state = self.get_state()  # Get the initial state.
            done = False

            while not done:
                action = self.choose_action(state)  # Choose an action.
                new_state, reward = self.take_action(action, state)  # Take the action and get the new state and reward.
                self.update_q_table(state, action, reward, new_state)  # Update the Q-table.
                state = new_state  # Update the state.

                # The episode is done when the time left is 0 or the budget is exhausted
                if self.total_time <= 0 or self.budget <= 0:
                    done = True

                    # Add a penalty if the budget is exhausted before the time is up
                    if self.budget <= 0 and self.total_time > 0:
                        reward -= self.penalty  # Adjust the penalty value as needed

            # Update the exploration rate at the end of the episode
            self.update_exploration_rate(episode)

    def get_optimal_policy(self):
        """
        Get the optimal policy after the Q-Learning agent has been trained.
        This function returns the optimal policy.
        """
        # The optimal policy is the action with the highest Q-value for each state.
        optimal_policy = np.argmax(self.q_table, axis=1)
        return optimal_policy

####################################################################################################################
##                                       Evaluation and Visualization                                              #
####################################################################################################################

    def evaluate_policy(self, policy):
        """
        Evaluate a given policy by calculating the total reward.

        policy: A list representing the policy (sequence of actions).

        This function returns the total reward for the given policy.
        """
        total_reward = 0
        self.current_city = policy[0]  # Start at the first city in the policy
        self.budget = self.initial_budget
        self.total_time = self.initial_total_time

        for i in range(len(policy) - 1):
            action = policy[i + 1]
            _, reward = self.take_action(action, self.get_state())
            total_reward += reward

        return total_reward

    def visualize_route(self, policy):
        """
        Visualize the optimal route on a map.

        policy: A list representing the policy (sequence of actions).

        This function plots the optimal route on a map.
        """
        # Create a GeoDataFrame to store the cities and their coordinates
        cities_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(city.geolocation[1], city.geolocation[0]) for city in self.cities]},
            crs="EPSG:4326")

        # Create a LineString object representing the route
        route_line = LineString([city.geolocation[::-1] for city in [self.cities[i] for i in policy]])

        # Create a GeoDataFrame to store the route
        route_gdf = gpd.GeoDataFrame({'geometry': [route_line]}, crs="EPSG:4326")

        # Plot the cities
        ax = cities_gdf.plot(marker='o', color='blue', markersize=50, figsize=(10, 10))

        # Plot the route
        route_gdf.plot(ax=ax, color='red')

        # Annotate the cities with their names
        for x, y, label in [(city.geolocation[1], city.geolocation[0], city.name) for city in self.cities]:
            ax.text(x, y, label, fontsize=12, ha='left')

        plt.title('Optimal Route')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
