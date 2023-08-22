import numpy as np
from qtrip_model import City, QLearningAgent

# Dataset
cities = [
    City("Tokyo", (35.682839, 139.759455), {"culture": 5, "food": 4, "nature": 3}, 10000000, 0),
    City("Kyoto", (35.011564, 135.768149), {"culture": 5, "food": 3, "nature": 4}, 1500000, 1),
    City("Osaka", (34.693738, 135.502165), {"culture": 4, "food": 5, "nature": 2}, 5000000, 2),
    City("Hiroshima", (34.385203, 132.455293), {"culture": 3, "food": 3, "nature": 3, "history": 5}, 1200000, 3),
    City("Sapporo", (43.061977, 141.354450), {"culture": 2, "food": 4, "nature": 5}, 2000000, 4),
    City("Yokohama", (35.443708, 139.638026), {"culture": 3, "food": 4, "nature": 3}, 3700000, 5),
    City("Nagoya", (35.181473, 136.906738), {"culture": 3, "food": 4, "nature": 2}, 2300000, 6),
    City("Fukuoka", (33.590355, 130.401716), {"culture": 3, "food": 4, "nature": 4}, 1500000, 7),
    City("Kobe", (34.691287, 135.183071), {"culture": 3, "food": 4, "nature": 3}, 1500000, 8),
    City("Kanazawa", (36.561325, 136.656205), {"culture": 4, "food": 3, "nature": 4}, 460000, 9),
    City("Sendai", (38.268215, 140.869355), {"culture": 3, "food": 3, "nature": 4}, 1070000, 10),
    City("Nara", (34.685087, 135.805000), {"culture": 5, "food": 3, "nature": 4}, 360000, 11),
    City("Hakodate", (41.768793, 140.728810), {"culture": 2, "food": 4, "nature": 5}, 270000, 12),
    City("Kumamoto", (32.803100, 130.707891), {"culture": 3, "food": 3, "nature": 4}, 740000, 13),
    City("Nagasaki", (32.750286, 129.877667), {"culture": 3, "food": 4, "nature": 4}, 410000, 14),
    City("Okinawa", (26.501301, 127.945404), {"culture": 2, "food": 5, "nature": 5}, 1430000, 15),
    City("Shizuoka", (34.975562, 138.382760), {"culture": 3, "food": 3, "nature": 4}, 700000, 16),
    City("Niigata", (37.916192, 139.036413), {"culture": 2, "food": 4, "nature": 4}, 800000, 17),
    City("Matsuyama", (33.839157, 132.765575), {"culture": 3, "food": 3, "nature": 4}, 510000, 18),
    City("Chiba", (35.605058, 140.123308), {"culture": 3, "food": 3, "nature": 3}, 970000, 19)
]

# Parameters
user_preferences = {"history": 4}
learning_rate = 0.005
discount_factor = 0.95
exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay = 0.995
num_episodes = 50000
total_time = 7
threshold = np.exp(-33)

# Create the agent
agent = QLearningAgent(learning_rate, discount_factor, exploration_rate, exploration_decay, num_episodes, cities, total_time, user_preferences, None, None)
agent.setup_environment()

# Model Training
it = 0
for episode in range(num_episodes):
    agent.total_time = total_time  # Reset the total time
    agent.current_city = np.random.choice(agent.num_cities)  # Start at a random city
    for day in range(total_time):
        state = agent.get_state(agent.current_city, day)  # Get the state for the current city and day
        action = agent.choose_action(state)  # Choose an action
        new_state, reward = agent.take_action(action, day)  # Take the action
        loss = agent.update_q_table(state, action, reward, new_state, day)  # Update the Q-table for the current day

        print('*--------------------------------------------------------------------*')
        print(f"QTripAgent | Episode {episode} | Iter {it}")
        print(f'Value MSE Loss: {loss}')
        print(f'Took action: {action}')
        print(f'State has been updated')
        print(f'FROM {state}')
        print(f'TO {new_state}')

        # Update the current city based on the new state
        agent.current_city = agent.state_to_index(new_state)[0]

        # Update the iteration index
        it += 1

    agent.exploration_rate *= exploration_decay

# Q* table display
print('*------------------------------------------------------*')
print('                      Q* table')
print(agent.q_table)

# Policy convergence
print('*--------------------------------------------------------------------*')
print('Converged to the following optimal policy')
policy = agent.get_optimal_policy()
print(policy)
print(f'Policy evaluation {agent.evaluate_policy(policy)}')
print('*--------------------------------------------------------------------*')

# Construct Itinerary
itinerary = [f"{cities[city_index].name}" for city_index, days in policy]
print('QTripAgent found the following itinerary')
day = 1
for city_index in policy:
    print(f'Day {day}: {cities[city_index[0]].name}')
    day += 1
print('*--------------------------------------------------------------------*')
print('      QTripAgent RL algorithm by https://github.com/basedryo')
print('*--------------------------------------------------------------------*')

# Visualize policy
agent.visualize_route(policy)
