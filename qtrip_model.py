import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString

'''
**DEV-NOTES (17/08/2023)**
    - Budget has been cancelled from being a component of the state, and from being a crucial part
      for action choice and reward calculation. The algorithm needs to work without taking in consideration
      the money factor (which will be a thing in the main real-time script), as accorded with the Q-Trip
      founders. The list of cities should be provided from the user preferences, model prior features
      and time.
**DEV-NOTES (21/08/2023)**
    - The model is finally able to converge by establishing a threshold on the MSE loss function during the
      model's setup (test.py). The model seems to choose a city stochastically (the reward function has to be
      carefully re-weighted), and the move_to_new_city() action is not even taken in consideration by an optimal
      policy PI_opt.
    - Need to check the desired_cities features since it is being incorrectly called in get_possible_actions()
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

# Example City class. Nation example: Let Japan(self.cities) = [Tokyo = City(...), Osaka = City(...), ...]
class City:
    """
    Example city class (vector components of the nation)

    name: name of the city
    coordinates: geolocation of the city (for Haversine formula)
    tags: useful tags for user_preferences city matching
    population: the city's population for model_prior_features
    """
    def __init__(self, name, coordinates, tags, population, index):
        self.name = name
        self.coordinates = coordinates
        self.geolocation = coordinates
        self.tags = tags
        self.population = population
        self.index = index

class QLearningAgent:

####################################################################################################################
#                                        Init and accessory methods                                                #
####################################################################################################################

    def __init__(self, learning_rate, discount_factor, exploration_rate, exploration_decay, num_episodes, cities, total_time, user_prompted_preferences, desired_cities, transportation_scores):
        """
        Initialize the Q-Learning Agent.

        learning_rate: The rate at which the Q-table is updated.
        discount_factor: The discount factor for future rewards.
        exploration_rate: The rate at which the agent chooses a random action for exploration.
        num_episodes: The number of episodes for the agent to learn from.
        cities: A list of City objects.
        initial_total_time: Store the initial total time
        total_time: The total time for the trip.
        user_preferences: The user's preferences for each city.
        desired_cities: The cities the user wants to visit. If None, the user has no preference.
        """

        # Model's initialization parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.cities = cities
        self.num_cities = len(cities)
        self.initial_total_time = total_time
        self.total_time = total_time
        self.user_prompted_preferences = user_prompted_preferences
        self.desired_cities = desired_cities if desired_cities is not None else list(range(self.num_cities))
        self.days_threshold = 4
        self.q_table = None
        self.penalty = None

        # Preferences for reward calculation (fixed)
        self.user_preferences = self.calculate_user_preferences()
        self.model_prior_features = self.calculate_model_prior_features()

        # Exploration rates and decayament
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01
        self.rates = []

    def setup_environment(self):
        """
        Initializes the Q-table with small random values between 0 and 0.01 to encourage exploration

        :return: initial Q-table
        """

        self.q_table = np.zeros((len(self.cities), 2, self.total_time))

    def update_exploration_rate(self, episode):
        """
        Decreases the exploration rate over time.

        :param episode: current episode (s,a,r,s')
        :return: updated exploration rate
        """

        # Linear decay
        decay_rate = 0.01
        self.exploration_rate -= decay_rate

        # Ensure that the exploration rate does not go below a minimum threshold
        min_exploration_rate = 0.1
        self.exploration_rate = max(self.exploration_rate, min_exploration_rate)

    def calculate_distance(self, city1, city2):
        """
        Given two cities, returns the distance between the two (Haversine formula)

        :param city1: the first city
        :param city2: the second city
        :return: distance
        """

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [city1.geolocation[0], city1.geolocation[1], city2.geolocation[0], city2.geolocation[1]])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371

        return c * r

    def calculate_model_prior_features(self) -> list:
        """
        Establishes a biased and priorly defined set of weights with respect to some city features (e.g. population)

        :return: model's preferences vector
        """

        populations = [city.population for city in self.cities]
        max_population = max(populations)
        return [population / max_population for population in populations]

    def calculate_user_preferences(self) -> list:
        """
        Establishes a set of weights with respect the matching of the prompted user preferences and the city features (e.g. culture, food)

        :return: user's preferences vector
        """

        user_preferences = []
        for city in self.cities:
            preference_score = sum([city.tags[tag] for tag in self.user_prompted_preferences if tag in city.tags])
            user_preferences.append(preference_score/10)
        return user_preferences

### Model's methods
    def get_state(self, current_city_index, current_day):
        """
        Give the index and day, returns the corresponding state
        :param current_city_index: index of the current city
        :param current_day: current day
        :return:
        """

        city_state = [0] * len(self.cities)
        city_state[current_city_index] = 1
        return city_state + [current_day]

    def state_to_index(self, state):
        """
        Given a state, returns the corresponding city index
        :param state: current state
        :return: index
        """
        city_state = state[:-1]
        time_state = state[-1]
        city_index = city_state.index(1)
        return city_index, time_state

    def get_possible_actions(self):
        """
        Returns a list of possible actions
        0: stay in the current state
        1: move to another state

        :return: list of actions
        """

        return [0, 1]

    def choose_action(self, state):
        """
        Chooses the action based on exploration rate defined odds

        :param state: current state
        :return: action
        """

        state_index, current_day = self.state_to_index(state)
        if np.random.rand() < self.exploration_rate:
            action = np.random.choice(self.get_possible_actions())
        else:
            action = np.argmax(self.q_table[state_index, :, current_day])
        return action

    def calculate_reward(self, current_city_index, new_city_index) -> float:
        """
        Calculates the reward of taking an action in a state and ending up on another state

        :param current_city_index: index of the current city
        :param new_city_index: index of the new city
        :return: reward
        """

        # Calculates the preferences parameter based on self.user_preferences and self.model_prior_features
        weight = [0.8, 0.2]
        if np.all(self.user_preferences == np.zeros(len(self.cities))):
            preferences = self.model_prior_features[current_city_index]
        else:
            preferences = weight[0] * self.user_preferences[current_city_index] + weight[1] * self.model_prior_features[current_city_index]

        # Distance between current city and new city
        distance = self.calculate_distance(self.cities[current_city_index], self.cities[new_city_index])

        # Scaling the moving cost by a factor (for example, 100)
        scaling_factor = 1000
        moving_cost = -distance / scaling_factor

        # Daily reward for staying in the current city
        daily_reward = preferences

        # Stay penalty
        stay_penalty = -0.999999999999942 if current_city_index == new_city_index else 0

        # Total reward for moving to a new city or staying in the current one
        reward = daily_reward + moving_cost if current_city_index != new_city_index else daily_reward + stay_penalty

        # Normalize the reward by the total time of the trip
        if self.total_time > 0:
            reward /= self.total_time

        # Apply a discount factor to the reward based on the distance between the current city and the new city.
        reward *= np.exp(-self.discount_factor * distance)  # Apply discounting based on distance

        return reward

    def move_to_new_city(self, current_day):
        """
        Moves from a city to another

        :param current_day: current day
        :return: new state, reward
        """

        current_state = self.get_state(self.current_city, current_day)
        current_state_index, _ = self.state_to_index(current_state)

        original_reward = self.q_table[current_state_index, 1, current_day]
        self.q_table[current_state_index, 1, current_day] = -1e10

        new_city_index = np.argmax(self.q_table[:, 1, current_day])

        self.q_table[current_state_index, 1, current_day] = original_reward

        reward = self.calculate_reward(self.current_city, new_city_index)
        new_state = self.get_state(new_city_index, current_day)

        return new_state, reward

    def stay_in_current_city(self, current_day):
        """
        Stays in the current city

        :param current_day: current day
        :return: new state, reward
        """

        reward = self.calculate_reward(self.current_city, self.current_city)
        new_state = self.get_state(self.current_city, current_day)
        return new_state, reward

    def take_action(self, action, current_day):
        """
        Performs an action

        :param action: selected action
        :param current_day: current day
        :return: performed action in the current day
        """

        if action == 1:  # Move to new city
            return self.move_to_new_city(current_day)
        else:  # Stay in current city
            return self.stay_in_current_city(current_day)

    def update_q_table(self, state, action, reward, new_state, current_day):
        """
        Updates the value of the Q-table

        :param state: current state
        :param action: selected action
        :param reward: corresponding reward
        :param new_state: new state
        :param current_day: current day
        :return: updated Q-table, loss
        """

        state_index, _ = self.state_to_index(state)
        new_state_index, _ = self.state_to_index(new_state)

        # Check if current_day + 1 is within bounds
        next_day = current_day + 1 if current_day + 1 < self.initial_total_time else current_day

        # Target Q-value
        target_q = reward + self.discount_factor * np.max(self.q_table[new_state_index, :, next_day])

        # Current Q-value
        current_q = self.q_table[state_index, action, current_day]

        # Calculate the MSE loss
        loss = (target_q - current_q) ** 2

        # Q-learning update using the target Q-value
        self.q_table[state_index, action, current_day] = (1 - self.learning_rate) * current_q + self.learning_rate * target_q

        return loss

    def find_best_city(self):
        """
        Given a time smaller than the threshold, finds the largest valued city

        :return: best city
        """

        best_city_index = np.argmax([self.calculate_reward(i, i) for i in range(self.num_cities)])
        return best_city_index

    def train(self):
        """
        Trains the model

        :return: trained model
        """

        if self.initial_total_time < self.days_threshold:
            # Find the best city
            best_city_index = self.find_best_city()

            # Fill the Q-table for the best city for all days
            self.q_table[:, 1, :] = -1e10  # Set a large negative value for moving to a new city
            self.q_table[best_city_index, 1, :] = 1e10  # Set a large positive value for staying in the best city
            return

        for episode in range(self.num_episodes):
            self.current_city = np.random.choice(self.num_cities)  # Start at a random city
            for day in range(self.initial_total_time):
                state = self.get_state(self.current_city, day)  # Get the state for the current city and day
                action = self.choose_action(state)  # Choose an action
                new_state, reward = self.take_action(action)  # Take the action
                new_city_index = self.state_to_index(new_state)[0]  # Update the current city based on the new state
                self.update_q_table(state, action, reward, new_state, day)  # Update the Q-table for the current day
                self.current_city = new_city_index  # Update the current city

                # Reduce the total time (assuming each action takes one day)
                self.total_time -= 1

                # Update the exploration rate if needed
                self.update_exploration_rate(episode)

    def get_optimal_policy(self):
        """
        Gets the optimal policy for the Q-table tensor and constructs the itinerary

        :return: optimal policy
        """

        if self.initial_total_time < self.days_threshold:
            best_city_index = self.find_best_city()
            return [(best_city_index, 1)] * self.initial_total_time

        itinerary = []
        for day in range(self.initial_total_time):
            city_index = np.argmax(np.max(self.q_table[:, :, day], axis=1))
            itinerary.append((city_index, 1))  # Each day corresponds to one city
        return itinerary

### Policy evaluation and visualization

    def evaluate_policy(self, policy):
        """
        Evaluates a given policy by calculating the total reward.

        :param policy: selected policy
        :return: policy evaluation
        """

        total_reward = 0
        self.current_city = policy[0][0]  # Start at the first city in the policy
        # self.budget = self.initial_budget
        self.total_time = self.initial_total_time

        for i in range(len(policy) - 1):
            action = policy[i + 1][1]  # Extracting the action from the tuple
            _, reward = self.take_action(action, i)  # Providing the current_day as i
            total_reward += reward

        return total_reward

    def visualize_route(self, policy):
        """
        Plots the optimal route on a map

        :param policy: selected policy
        :return: map plot
        """

        # Create a GeoDataFrame to store the cities and their coordinates
        cities_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(city.geolocation[1], city.geolocation[0]) for city in self.cities]},
            crs="EPSG:4326")

        # Extract city indices from the policy and get corresponding cities
        city_indices = [item[0] for item in policy]  # Extracting city indices
        selected_cities = [self.cities[i] for i in city_indices]

        # Create a LineString object representing the route
        route_line = LineString([city.geolocation[::-1] for city in selected_cities])

        # Create a GeoDataFrame to store the route
        route_gdf = gpd.GeoDataFrame({'geometry': [route_line]}, crs="EPSG:4326")

        # Plot the cities
        ax = cities_gdf.plot(marker='o', color='blue', markersize=50, figsize=(10, 10))

        # Plot the route
        route_gdf.plot(ax=ax, color='red')

        # Annotate the cities with their names
        for x, y, label in [(city.geolocation[1], city.geolocation[0], city.name) for city in self.cities]:
            ax.text(x, y, label, fontsize=12, ha='left')

        plt.title('QTripAgent Optimal Route')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
