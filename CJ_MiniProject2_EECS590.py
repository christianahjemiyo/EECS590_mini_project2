from CJ_markovprocess import MarkovProcess
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

print("=== TOP OF FILE STARTED ===")
class MDP(MarkovProcess):
    """
    This is our main Solver class.
    It inherits the 'physics' requirements from MarkovProcess and adds
    the 'math' (Value Iteration, Policy Iteration).
    """

    def __init__(self, gamma=0.9):
        self.gamma = gamma  # Discount factor
        self.V = {}  # State Value Function: V(s)
        self.Q = {}  # Action-Value Function: Q(s, a)
        self.policy = {}  # The Policy: pi(s)
        self.initial_states = []  # Initial Measure (mu)

    # ---------------------------------------------------------
    # HELPER METHOD (Static)
    # ---------------------------------------------------------
    @staticmethod
    def _det(state):
        """
        Helper to handle deterministic transitions cleanly.
        Usage: return self._det((0, 1)) or MDP._det((0, 1))
        """
        return {state: 1.0}

    def set_initial_states(self, states):
        """Define the starting distribution mu."""
        self.initial_states = states

    # -------------------------------------------------------------------------
    # ABSTRACT METHOD (This is required because MarkovProcess didn't have it)
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_available_actions(self, state):
        """
        We must implement this in our specific problems (Drone, Robot, etc.).
        Returns a list of valid actions (e.g., ['up', 'down', 'left', 'right']).
        """
        pass

    # ---------------------------------------------------------
    # ALGORITHM 1: Policy Iteration
    # ---------------------------------------------------------
    def policy_iteration(self, theta=1e-6):
        """
        Loop:
        1. Policy Evaluation: Calculate V(s) for current policy
        2. Policy Improvement: Update pi(s) to be greedy w.r.t V(s)
        """
        is_policy_stable = False
        i = 0

        # Initialize a random policy if one doesn't exist yet
        if not self.policy:
            self.initialize_random_policy()

        while not is_policy_stable:
            i += 1
            # Step 1: Evaluate
            self.policy_evaluation(theta)

            # Step 2: Improve
            is_policy_stable = self.policy_improvement()
            # print(f"Policy Iteration {i}: Stable? {is_policy_stable}")

    def initialize_random_policy(self):
        """Sets initial policy to random valid actions."""
        for s in self.get_states():
            actions = self.get_available_actions(s)
            if actions:
                self.policy[s] = np.random.choice(actions)

    def policy_evaluation(self, theta=1e-6):
        """
        Iteratively updates V(s) until the change is smaller than theta.
        V(s) = Sum_s' [ P(s'|s, pi(s)) * (R + gamma * V(s')) ]
        """
        while True:
            delta = 0
            for s in self.get_states():
                v_old = self.V.get(s, 0)

                # If state has no actions (terminal?), value is 0
                if s not in self.policy:
                    continue

                a = self.policy[s]

                # We expect get_transition_prob to return a dictionary {next_s: prob}
                v_new = 0
                transitions = self.get_transition_prob(s, a)

                for next_s, prob in transitions.items():
                    r = self.get_reward(s, a, next_s)
                    v_new += prob * (r + self.gamma * self.V.get(next_s, 0))

                self.V[s] = v_new
                delta = max(delta, abs(v_old - v_new))

            if delta < theta:
                break

    def policy_improvement(self):
        """
        Updates policy to be greedy.
        If pi(s) was already the best action, returns True (Stable).
        """
        policy_stable = True

        for s in self.get_states():
            old_action = self.policy.get(s)

            # Find the action that maximizes Q(s,a)
            best_action_val = -float('inf')
            best_action = None

            for a in self.get_available_actions(s):
                # Calculate Q(s, a)
                q_val = 0
                transitions = self.get_transition_prob(s, a)

                for next_s, prob in transitions.items():
                    r = self.get_reward(s, a, next_s)
                    q_val += prob * (r + self.gamma * self.V.get(next_s, 0))

                if q_val > best_action_val:
                    best_action_val = q_val
                    best_action = a

            self.policy[s] = best_action

            if old_action != best_action:
                policy_stable = False

        return policy_stable

    # ---------------------------------------------------------
    # ALGORITHM 2: Value Iteration
    # ---------------------------------------------------------
    def value_iteration(self, theta=1e-6):
        """
        Finds optimal V(s) by taking the max over actions in every step.
        V_k+1(s) = max_a [ Sum_s' P(s'|s,a) * (R + gamma * V_k(s')) ]
        """
        while True:
            delta = 0
            for s in self.get_states():
                v_old = self.V.get(s, 0)
                max_q_val = -float('inf')

                # If no actions available (terminal), V is 0
                possible_actions = self.get_available_actions(s)
                if not possible_actions:
                    self.V[s] = 0
                    continue

                for a in possible_actions:
                    q_val = 0
                    transitions = self.get_transition_prob(s, a)

                    for next_s, prob in transitions.items():
                        r = self.get_reward(s, a, next_s)
                        q_val += prob * (r + self.gamma * self.V.get(next_s, 0))

                    if q_val > max_q_val:
                        max_q_val = q_val

                self.V[s] = max_q_val
                delta = max(delta, abs(v_old - self.V[s]))

            if delta < theta:
                break

        # Derive the optimal policy from the final V(s)
        self.extract_policy()

    def extract_policy(self):
        """Fills self.policy with the best action for each state."""
        for s in self.get_states():
            best_action_val = -float('inf')
            best_action = None

            possible_actions = self.get_available_actions(s)
            if not possible_actions:
                continue

            for a in possible_actions:
                q_val = 0
                transitions = self.get_transition_prob(s, a)

                for next_s, prob in transitions.items():
                    r = self.get_reward(s, a, next_s)
                    q_val += prob * (r + self.gamma * self.V.get(next_s, 0))

                if q_val > best_action_val:
                    best_action_val = q_val
                    best_action = a

            self.policy[s] = best_action


# Problem 1: Navigating A Windy Chasm

# ---------------------------------------------------------
#  The Windy Chasm Logic Class
# ---------------------------------------------------------

class WindyChasmMDP(MDP):
    def __init__(self, p_center=0.5, p2=0.1, R_goal=100, r_crash=-100, gamma=0.9):
        super().__init__(gamma)

        # Environment Constants
        self.grid_rows = 20
        self.grid_cols = 7
        self.p_center = p_center  # This is B
        self.p2 = p2  # The probability of a secondary push
        self.R_goal = R_goal
        self.r_crash = r_crash

        # Special Terminal States
        self.GOAL = "GOAL"
        self.CRASH = "CRASH"

        # Define the state space
        self.all_states = []

        # 1. Add all grid coordinates (i, j)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.all_states.append((i, j))

        # 2. Add terminal states
        self.all_states.append(self.GOAL)
        self.all_states.append(self.CRASH)

        # Define Initial Measure (mu)
        self.set_initial_states([(0, 3)])

    def get_states(self):
        return self.all_states

    def get_available_actions(self, state):
        # If we are in a terminal state, no actions available
        if state == self.GOAL or state == self.CRASH:
            return []

        # The drone can always try to move Forward, Left, or Right
        return ['F', 'L', 'R']

    def _calculate_wind_prob(self, j_col):
        """
        Calculates the probability parameter p based on the column j.
        Formula: p(j) = B ^ (1 / (1 + (j-3)^2))
        """
        if j_col == 3:
            return self.p_center
        else:
            # Calculate the exponent E(j)
            E = 1.0 / (1.0 + (j_col - 3) ** 2)

            # Apply exponentiation: p(j) = B ^ E(j)
            return self.p_center ** E

    def get_transition_prob(self, state, action):
        # If already terminal, stay there with prob 1
        if state == self.GOAL or state == self.CRASH:
            return {state: 1.0}

        curr_i, curr_j = state

        # 1. Deterministic Action Movement
        next_i = curr_i
        next_j = curr_j

        if action == 'F':
            next_i += 1
        elif action == 'L':
            next_j -= 1
        elif action == 'R':
            next_j += 1

        # Crash if we hit the chasm walls (j <= 0 or j >= 6 for 7 columns)
        if next_j <= 0 or next_j >= self.grid_cols - 1:
            return {self.CRASH: 1.0}

        # Check if moving past the end of the grid
        if next_i >= self.grid_rows:
            return {self.CRASH: 1.0}

        # 2. Apply Wind Dynamics
        p_wind = self._calculate_wind_prob(next_j)

        prob_push_1 = p_wind
        prob_push_2 = (1 - p_wind) * self.p2
        prob_stay = (1 - p_wind) * (1 - self.p2)

        transitions = {}

        def add_outcome(i, j, prob):
            """
            IMPORTANT FIX:
            - Crash happens if we land on walls: j <= 0 or j >= 6 (for 7 columns).
            - Crash should override goal on the walls.
            - Goal happens when i == 19 in the safe interior.
            """
            # Crash first (walls / out-of-bounds)
            if j <= 0 or j >= self.grid_cols - 1:
                transitions[self.CRASH] = transitions.get(self.CRASH, 0) + prob
            # Goal next (only if not crashed)
            elif i == 19:
                transitions[self.GOAL] = transitions.get(self.GOAL, 0) + prob
            else:
                s = (i, j)
                transitions[s] = transitions.get(s, 0) + prob

        # 1. Stay
        add_outcome(next_i, next_j, prob_stay)

        # 2. Push +/- 1
        if prob_push_1 > 0:
            add_outcome(next_i, next_j + 1, prob_push_1 * 0.5)
            add_outcome(next_i, next_j - 1, prob_push_1 * 0.5)

        # 3. Push +/- 2
        if prob_push_2 > 0:
            add_outcome(next_i, next_j + 2, prob_push_2 * 0.5)
            add_outcome(next_i, next_j - 2, prob_push_2 * 0.5)

        return transitions

    def get_reward(self, state, action, next_state):
        if next_state == self.GOAL:
            return self.R_goal
        elif next_state == self.CRASH:
            return self.r_crash
        else:
            return -1


# ---------------------------------------------------------
# The Gymnasium Wrapper (Visualization)
# ---------------------------------------------------------
class WindyChasmGym(gym.Env):
    def __init__(self, mdp_instance):
        super().__init__()
        self.mdp = mdp_instance
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: 'F', 1: 'L', 2: 'R'}
        self.observation_space = spaces.MultiDiscrete([mdp_instance.grid_rows, mdp_instance.grid_cols])
        self.state = None
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = random.choice(self.mdp.initial_states)
        return self.state, {}

    def step(self, action):
        action_str = self.action_map[action]
        transitions = self.mdp.get_transition_prob(self.state, action_str)

        states_list = list(transitions.keys())
        probs_list = list(transitions.values())

        next_state_list = random.choices(states_list, weights=probs_list, k=1)
        next_state = next_state_list[0]

        reward = self.mdp.get_reward(self.state, action_str, next_state)

        terminated = (next_state == "GOAL" or next_state == "CRASH")

        self.state = next_state
        return self.state, reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.mdp.grid_rows - 0.5)
        self.ax.set_ylim(-0.5, self.mdp.grid_cols - 0.5)
        self.ax.set_xticks(np.arange(self.mdp.grid_rows))
        self.ax.set_yticks(np.arange(self.mdp.grid_cols))
        self.ax.grid(which='both')

        # Draw Goal (Green) and Crash (Red) Zones
        for j in range(self.mdp.grid_cols):
            self.ax.add_patch(plt.Rectangle((19 - 0.4, j - 0.4), 0.8, 0.8, color='green', alpha=0.3))
            self.ax.add_patch(plt.Rectangle((0 - 0.4, j - 0.4), 0.8, 0.8, color='red', alpha=0.1))
            self.ax.add_patch(plt.Rectangle((19 - 0.4, 6 - 0.4), 0.8, 0.8, color='red', alpha=0.1))

        # Draw Drone
        if self.state != "GOAL" and self.state != "CRASH":
            i, j = self.state
            opt_action = self.mdp.policy.get(self.state, '?')
            self.ax.plot(i, j, 'bo', markersize=15)
            self.ax.text(i, j + 0.3, opt_action, ha='center', fontsize=12,
                         fontweight='bold', color='blue')
        elif self.state == "GOAL":
            self.ax.text(9, 3, "GOAL!", color='green', fontsize=20, ha='center')
        elif self.state == "CRASH":
            self.ax.text(9, 3, "CRASHED!", color='red', fontsize=20, ha='center')
        plt.pause(0.1)


# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Windy Chasm MDP...")
    drone_logic = WindyChasmMDP(p_center=0.5, p2=0.1, R_goal=100, r_crash=-100)

    # Optional sanity check (remove if you don't want extra output)
    print("Rows, Cols:", drone_logic.grid_rows, drone_logic.grid_cols)
    print("Total states (including GOAL/CRASH):", len(drone_logic.get_states()))

    print("Running Value Iteration (Solving for Optimal Policy)...")
    drone_logic.value_iteration(theta=1e-4)
    print("Solving Complete.")

    print("Initializing Gymnasium Environment...")
    env = WindyChasmGym(drone_logic)

    state, info = env.reset()
    plt.ion()
    plt.show()

    print("Starting Simulation (Close the window to stop)...")
    step_count = 0
    terminated = False

    while not terminated:
        opt_str = drone_logic.policy.get(state, 'F')

        gym_act = 0
        if opt_str == 'L':
            gym_act = 1
        elif opt_str == 'R':
            gym_act = 2

        state, reward, terminated, truncated, info = env.step(gym_act)
        step_count += 1
        env.render()

    print(f"Episode finished in {step_count} steps. Result: {state}")
    plt.ioff()
    plt.show()

