import random
import matplotlib.pyplot as plt

step_size=0.5
discount=1
epsilon=0.1

a_list=[(0,1),(1,0),(0,-1),(-1,0)]
all_s= [(r,c) for r in range(12) for c in range(4)]
cliff_s=[(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0)]
start_s=(0,0)
goal_s=(11,0)


def get_next_state(state,action):
    return (state[0]+action[0],state[1]+action[1])

def get_immediate_reward(state,action):
    (x,y)=get_next_state(state, action)
    if (x,y) in cliff_s:
        reward=-100
    elif (x,y)==goal_s:
        reward=0
    else:
        reward=-1
    return reward


def get_action(state):
    possible_a = []
    for action in a_list:
        if (state[0] + action[0], state[1] + action[1]) in all_s:
            possible_a.append(action)
        else:
            pass
    # max_pair = max(((state, a) for a in possible_a), key=lambda pair: Q[pair])
    # best_action = max_pair[1]
    e = random.uniform(0, 1)
    if e <= epsilon:  # for epsilon=0.1
        current_a = random.choice(possible_a)
    else:
        current_a = max(possible_a, key=lambda a: Q[(state, a)])
        #  current_a = random.choice([a for a in possible_a if Q[(state, a)] == max(Q[(state, a)] for a in possible_a)])
    return current_a

# Q = {(s, a): random.uniform(-0.01, 0.01) for s in all_s for a in a_list}
Q = {(s, a): 0 for s in all_s for a in a_list} #Q dictionary initialized to 0
reward_sum_dictionary={}

for episode in range(600):
    # epsilon = max(0.01, 0.1 * (0.99 ** episode))

    current_s = start_s
    reward_sum=0
    max_steps =200  # Prevent infinite loops
    step_count = 0
    current_a=get_action(current_s)
    # if episode % 50 == 0:
    #     print(f"🔍 Episode {episode}: Q-values for (1,1) → {Q[((0, 2), (0,1))]}")

    while step_count < max_steps:
        current_Q_value = Q[(current_s, current_a)]
        immediate_r = get_immediate_reward(current_s, current_a)
        reward_sum += immediate_r
        next_s = get_next_state(current_s, current_a)


        if next_s in cliff_s or next_s==goal_s:
            Q[(current_s, current_a)] = Q[(current_s, current_a)] + step_size * (immediate_r - Q[(current_s, current_a)])
            break
        else:
            pass


        next_a=get_action(next_s)
        next_Q_value=Q[(next_s, next_a)]
        Q[(current_s, current_a)] = Q[(current_s, current_a)] + step_size * (immediate_r + discount * Q[(next_s, next_a)] - Q[(current_s, current_a)])  # SARSA UPDATE
        current_s = next_s
        current_a= next_a
        step_count += 1

    reward_sum_dictionary[episode] = reward_sum



def get_trajectory(Q, start_s, goal_s, cliff_s, all_s, a_list):
    trajectory = []  # This will store the sequence of (state, action) pairs
    current_s = start_s  # Start from the initial state
    while current_s != goal_s:
        # Get valid actions for the current state
        possible_a = []
        for action in a_list:
            if (current_s[0] + action[0], current_s[1] + action[1]) in all_s:
                possible_a.append(action)
            else:
                pass

        best_a = max(possible_a, key=lambda a: Q[(current_s, a)])
        trajectory.append((current_s, best_a))  # Add (state, action) to the trajectory
        next_s = get_next_state(current_s, best_a)  # Move to the next state
        current_s = next_s

    return trajectory

def visualize_trajectory(trajectory, all_s, cliff_s, goal_s):
    grid = [[' ' for _ in range(4)] for _ in range(12)]

    # Mark cliffs
    for (r, c) in cliff_s:
        grid[r][c] = 'C'

    # Mark goal
    grid[goal_s[0]][goal_s[1]] = 'G'
    grid[start_s[0]][start_s[1]]='S'

    # Mark trajectory
    for (r, c), _ in trajectory:
        if grid[r][c] == ' ':
            grid[r][c] = '.'

    # Print the grid
    for row in grid:
        print(' '.join(row))


# Visualize the trajectory
trajectory=get_trajectory(Q, start_s, goal_s, cliff_s, all_s, a_list)
visualize_trajectory(trajectory, all_s, cliff_s, goal_s)


episodes = list(reward_sum_dictionary.keys())  # X-axis (episodes)
total_rewards = list(reward_sum_dictionary.values())  # Y-axis (total rewards)
# print("Episodes:", episodes)
# print("Total Rewards:", total_rewards)
# Plotting

plt.plot(episodes, total_rewards, linestyle='-', color='b', label='Reward Sum')

# Labels and title
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Rewards per Episode")
plt.legend()
plt.grid(True)

plt.show()