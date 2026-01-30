from CJ_markovprocess import MarkovProcess
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random


# ==========================================================
# 1. THE MDP SOLVER
# ==========================================================
class MDP(MarkovProcess):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.V = {}
        self.Q = {}
        self.policy = {}
        self.initial_states = []

    @staticmethod
    def _det(state):
        return {state: 1.0}

    def set_initial_states(self, states):
        self.initial_states = states

    @abstractmethod
    def get_available_actions(self, state):
        pass

    def value_iteration(self, theta=1e-6):
        while True:
            delta = 0
            for s in self.get_states():
                v_old = self.V.get(s, 0)
                max_q_val = -float('inf')
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
        self.extract_policy()

    def extract_policy(self):
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


# ==========================================================
# 2. THE FINE-GRAINED DRONE ENVIRONMENT
# ==========================================================
class WindyChasmFineMDP(MDP):
    def __init__(self, dx=0.05, dy=1, p_center=0.8, p2=0.1, R_goal=100, r_crash=-100, gamma=0.9):
        super().__init__(gamma)

        # --- 1. RESOLUTION PARAMETERS ---
        self.dx = dx  # Horizontal step size (0.05 instead of 1)
        self.dy = dy  # Vertical step size (kept at 1)

        self.max_x = 19.0
        self.max_y = 6.0

        # --- 2. PHYSICS PARAMETERS ---
        self.p_center = p_center
        self.p2 = p2
        self.R_goal = R_goal
        self.r_crash = r_crash

        # --- 3. STATE SPACE GENERATION ---
        self.GOAL = "GOAL"
        self.CRASH = "CRASH"
        self.all_states = []

        curr_x = 0.0
        while curr_x <= self.max_x:
            for j in range(7):
                # Round to avoid float precision errors in dictionary keys
                self.all_states.append((round(curr_x, 4), j))
            curr_x += self.dx

        self.all_states.append(self.GOAL)
        self.all_states.append(self.CRASH)

        self.set_initial_states([(0.0, 3)])

    def get_states(self):
        return self.all_states

    def get_available_actions(self, state):
        if state == self.GOAL or state == self.CRASH:
            return []
        return ['F', 'L', 'R']

    def _calculate_wind_prob(self, y_col):
        """Wind probability depends on vertical position (same as before)."""
        if y_col == 3:
            return self.p_center
        else:
            E = 1.0 / (1.0 + (y_col - 3) ** 2)
            return self.p_center ** E

    def get_transition_prob(self, state, action):
        if state == self.GOAL or state == self.CRASH:
            return {state: 1.0}

        curr_x, curr_y = state

        # 1. Apply Movement (Interpolation)
        next_x = curr_x
        next_y = curr_y

        if action == 'F':
            next_x += self.dx  # Tiny step
        elif action == 'L':
            next_y -= self.dy
        elif action == 'R':
            next_y += self.dy

        # 2. Bounds Check
        if next_y < 0 or next_y >= 7:
            return {self.CRASH: 1.0}

        # FIX: Only crash if we go PAST the max_x (> 19.0)
        # If we land exactly at 19.0, it is a goal.
        if next_x > self.max_x:
            return {self.CRASH: 1.0}

        # 3. Apply Wind Dynamics
        p_wind = self._calculate_wind_prob(next_y)

        prob_push_1 = p_wind
        prob_push_2 = (1 - p_wind) * self.p2
        prob_stay = (1 - p_wind) * (1 - self.p2)

        transitions = {}

        def add_outcome(x, y, prob):
            # Check Goal: Land ON or PAST the Goal line (19.0)
            # This logic now works because we removed the early return above
            if x >= self.max_x:
                transitions[self.GOAL] = transitions.get(self.GOAL, 0) + prob
            # Check Crash: Land ON or PAST the walls
            elif y < 0 or y >= 7:
                transitions[self.CRASH] = transitions.get(self.CRASH, 0) + prob
            else:
                s = (round(x, 4), y)
                transitions[s] = transitions.get(s, 0) + prob

        add_outcome(next_x, next_y, prob_stay)
        if prob_push_1 > 0:
            add_outcome(next_x, next_y + 1, prob_push_1 * 0.5)
            add_outcome(next_x, next_y - 1, prob_push_1 * 0.5)
        if prob_push_2 > 0:
            add_outcome(next_x, next_y + 2, prob_push_2 * 0.5)
            add_outcome(next_x, next_y - 2, prob_push_2 * 0.5)

        return transitions

    def get_reward(self, state, action, next_state):
        if next_state == self.GOAL:
            return self.R_goal
        elif next_state == self.CRASH:
            return self.r_crash
        else:
            return -1 * self.dx

        # ==========================================================


# 3. VISUALIZERS
# ==========================================================

def visualize_fine_policy_map(drone_mdp):
    """Displays the fine-grained policy as a static heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    dx = drone_mdp.dx
    x_coords = np.arange(0.0, 19.0 + dx, dx)
    y_coords = np.arange(7)

    V_grid = np.full((len(y_coords), len(x_coords)), np.nan)
    U = np.zeros((len(y_coords), len(x_coords)))
    V_dir = np.zeros((len(y_coords), len(x_coords)))

    for s in drone_mdp.get_states():
        if isinstance(s, str): continue
        x, y = s
        val = drone_mdp.V.get(s, 0)
        action = drone_mdp.policy.get(s, None)

        col_idx = int(round(x / dx))
        row_idx = int(y)

        V_grid[row_idx, col_idx] = val
        if action == 'F':
            U[row_idx, col_idx] = 1
            V_dir[row_idx, col_idx] = 0
        elif action == 'R':
            U[row_idx, col_idx] = 0
            V_dir[row_idx, col_idx] = 1
        elif action == 'L':
            U[row_idx, col_idx] = 0
            V_dir[row_idx, col_idx] = -1

    fig, ax = plt.subplots(figsize=(40, 6))
    c = ax.imshow(V_grid, cmap='viridis', origin='lower', aspect='equal',
                  extent=[0, 19, -0.5, 6.5])

    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
    ax.quiver(X_mesh, Y_mesh, U, V_dir, color='white', pivot='mid', scale=25)

    ax.set_title("Fine-Grained Policy (dx=0.05)")
    ax.set_xlabel("Physical Distance (i)")
    ax.set_ylabel("Width (j)")
    ax.set_xticks(np.arange(0, 20, 2))
    fig.colorbar(c, ax=ax, label="Value V(s)")
    plt.show()


class WindyChasmFineGym(gym.Env):
    def __init__(self, mdp_instance):
        super().__init__()
        self.mdp = mdp_instance
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: 'F', 1: 'L', 2: 'R'}
        self.observation_space = spaces.Box(low=np.array([0.0, 0]), high=np.array([19.0, 6]), dtype=np.float32)
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
        self.ax.set_xlim(-0.5, 19.5)
        self.ax.set_ylim(-0.5, 6.5)
        self.ax.set_xticks(np.arange(0, 20, 2))
        self.ax.set_yticks(np.arange(7))
        self.ax.grid(which='both')
        self.ax.set_title(f"Fine-Grained Sim - State: {self.state}")

        for j in range(7):
            self.ax.add_patch(plt.Rectangle((19 - 0.4, j - 0.4), 0.8, 0.8, color='green', alpha=0.3))
            self.ax.add_patch(plt.Rectangle((0 - 0.4, j - 0.4), 0.8, 0.8, color='red', alpha=0.1))

        if self.state != "GOAL" and self.state != "CRASH":
            x, y = self.state
            opt_action = self.mdp.policy.get(self.state, '?')
            self.ax.plot(x, y, 'bo', markersize=15)
            self.ax.text(x, y + 0.3, opt_action, ha='center', fontsize=12, fontweight='bold', color='blue')
        elif self.state == "GOAL":
            self.ax.text(9.5, 3, "GOAL!", color='green', fontsize=20, ha='center')
        elif self.state == "CRASH":
            self.ax.text(9.5, 3, "CRASHED!", color='red', fontsize=20, ha='center')
        plt.pause(0.01)


# ==========================================================
# 4. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("=== FINE-GRAINED WINDY CHASM (dx=0.05) ===")

    drone_fine = WindyChasmFineMDP(dx=0.05, dy=1, p_center=0.8, p2=0.1, R_goal=100, r_crash=-100)

    print(f"States: {len(drone_fine.get_states())} (Expected ~2660)")
    print("Solving Fine-Grained MDP...")

    drone_fine.value_iteration(theta=1e-4)
    print("Solving Complete.")

    print("Displaying Policy Map (Wide Plot)...")
    visualize_fine_policy_map(drone_fine)
    print("Close Policy Map to start Simulation.")

    env = WindyChasmFineGym(drone_fine)
    state, info = env.reset()
    plt.ion()
    plt.show()

    print("Starting Fine-Grained Simulation...")
    step_count = 0
    terminated = Falsefrom CJ_markovprocess import MarkovProcess
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random


# ==========================================================
# 1. THE MDP SOLVER
# ==========================================================
class MDP(MarkovProcess):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.V = {}
        self.Q = {}
        self.policy = {}
        self.initial_states = []

    @staticmethod
    def _det(state):
        return {state: 1.0}

    def set_initial_states(self, states):
        self.initial_states = states

    @abstractmethod
    def get_available_actions(self, state):
        pass

    def value_iteration(self, theta=1e-6):
        while True:
            delta = 0
            for s in self.get_states():
                v_old = self.V.get(s, 0)
                max_q_val = -float('inf')
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
        self.extract_policy()

    def extract_policy(self):
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


# ==========================================================
# 2. THE FINE-GRAINED DRONE ENVIRONMENT
# ==========================================================
class WindyChasmFineMDP(MDP):
    def __init__(self, dx=0.05, dy=1, p_center=0.8, p2=0.1, R_goal=100, r_crash=-100, gamma=0.9):
        super().__init__(gamma)

        # --- 1. RESOLUTION PARAMETERS ---
        self.dx = dx  # Horizontal step size (0.05 instead of 1)
        self.dy = dy  # Vertical step size (kept at 1)

        self.max_x = 19.0
        self.max_y = 6.0

        # --- 2. PHYSICS PARAMETERS ---
        self.p_center = p_center
        self.p2 = p2
        self.R_goal = R_goal
        self.r_crash = r_crash

        # --- 3. STATE SPACE GENERATION ---
        self.GOAL = "GOAL"
        self.CRASH = "CRASH"
        self.all_states = []

        curr_x = 0.0
        while curr_x <= self.max_x:
            for j in range(7):
                # Round to avoid float precision errors in dictionary keys
                self.all_states.append((round(curr_x, 4), j))
            curr_x += self.dx

        self.all_states.append(self.GOAL)
        self.all_states.append(self.CRASH)

        self.set_initial_states([(0.0, 3)])

    def get_states(self):
        return self.all_states

    def get_available_actions(self, state):
        if state == self.GOAL or state == self.CRASH:
            return []
        return ['F', 'L', 'R']

    def _calculate_wind_prob(self, y_col):
        """Wind probability depends on vertical position (same as before)."""
        if y_col == 3:
            return self.p_center
        else:
            E = 1.0 / (1.0 + (y_col - 3) ** 2)
            return self.p_center ** E

    def get_transition_prob(self, state, action):
        if state == self.GOAL or state == self.CRASH:
            return {state: 1.0}

        curr_x, curr_y = state

        # 1. Apply Movement (fine-grained step)
        next_x = curr_x
        next_y = curr_y

        if action == 'F':
            next_x += self.dx  # Fine-grained step forward
        elif action == 'L':
            next_y -= self.dy
        elif action == 'R':
            next_y += self.dy

        # 2. Check vertical bounds (crash immediately if out of bounds)
        if next_y < 0 or next_y >= 7:
            return {self.CRASH: 1.0}

        # 3. Apply Wind Dynamics
        p_wind = self._calculate_wind_prob(next_y)

        prob_push_1 = p_wind
        prob_push_2 = (1 - p_wind) * self.p2
        prob_stay = (1 - p_wind) * (1 - self.p2)

        transitions = {}

        def add_outcome(x, y, prob):
            # Check Goal: Reached or passed the goal line (x >= 19.0)
            if x >= self.max_x:
                transitions[self.GOAL] = transitions.get(self.GOAL, 0) + prob
            # Check Crash: Hit the vertical walls
            elif y < 0 or y >= 7:
                transitions[self.CRASH] = transitions.get(self.CRASH, 0) + prob
            else:
                s = (round(x, 4), y)
                transitions[s] = transitions.get(s, 0) + prob

        # Add all possible wind outcomes
        add_outcome(next_x, next_y, prob_stay)
        if prob_push_1 > 0:
            add_outcome(next_x, next_y + 1, prob_push_1 * 0.5)
            add_outcome(next_x, next_y - 1, prob_push_1 * 0.5)
        if prob_push_2 > 0:
            add_outcome(next_x, next_y + 2, prob_push_2 * 0.5)
            add_outcome(next_x, next_y - 2, prob_push_2 * 0.5)

        return transitions

    def get_reward(self, state, action, next_state):
        if next_state == self.GOAL:
            return self.R_goal
        elif next_state == self.CRASH:
            return self.r_crash
        else:
            # Fine-grained step cost scaled by dx
            return -1 * self.dx


# ==========================================================
# 3. VISUALIZERS
# ==========================================================

def visualize_fine_policy_map(drone_mdp):
    """Displays the fine-grained policy as a static heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    dx = drone_mdp.dx
    x_coords = np.arange(0.0, 19.0 + dx, dx)
    y_coords = np.arange(7)

    V_grid = np.full((len(y_coords), len(x_coords)), np.nan)
    U = np.zeros((len(y_coords), len(x_coords)))
    V_dir = np.zeros((len(y_coords), len(x_coords)))

    for s in drone_mdp.get_states():
        if isinstance(s, str): continue
        x, y = s
        val = drone_mdp.V.get(s, 0)
        action = drone_mdp.policy.get(s, None)

        col_idx = int(round(x / dx))
        row_idx = int(y)

        V_grid[row_idx, col_idx] = val
        if action == 'F':
            U[row_idx, col_idx] = 1
            V_dir[row_idx, col_idx] = 0
        elif action == 'R':
            U[row_idx, col_idx] = 0
            V_dir[row_idx, col_idx] = 1
        elif action == 'L':
            U[row_idx, col_idx] = 0
            V_dir[row_idx, col_idx] = -1

    fig, ax = plt.subplots(figsize=(40, 6))
    c = ax.imshow(V_grid, cmap='viridis', origin='lower', aspect='equal',
                  extent=[0, 19, -0.5, 6.5])

    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
    ax.quiver(X_mesh, Y_mesh, U, V_dir, color='white', pivot='mid', scale=25)

    ax.set_title("Fine-Grained Policy (dx=0.05)")
    ax.set_xlabel("Physical Distance (i)")
    ax.set_ylabel("Width (j)")
    ax.set_xticks(np.arange(0, 20, 2))
    fig.colorbar(c, ax=ax, label="Value V(s)")
    plt.show()


class WindyChasmFineGym(gym.Env):
    def __init__(self, mdp_instance):
        super().__init__()
        self.mdp = mdp_instance
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: 'F', 1: 'L', 2: 'R'}
        self.observation_space = spaces.Box(low=np.array([0.0, 0]), high=np.array([19.0, 6]), dtype=np.float32)
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
        self.ax.set_xlim(-0.5, 19.5)
        self.ax.set_ylim(-0.5, 6.5)
        self.ax.set_xticks(np.arange(0, 20, 2))
        self.ax.set_yticks(np.arange(7))
        self.ax.grid(which='both')
        self.ax.set_title(f"Fine-Grained Sim - State: {self.state}")

        for j in range(7):
            self.ax.add_patch(plt.Rectangle((19 - 0.4, j - 0.4), 0.8, 0.8, color='green', alpha=0.3))
            self.ax.add_patch(plt.Rectangle((0 - 0.4, j - 0.4), 0.8, 0.8, color='red', alpha=0.1))

        if self.state != "GOAL" and self.state != "CRASH":
            x, y = self.state
            opt_action = self.mdp.policy.get(self.state, '?')
            self.ax.plot(x, y, 'bo', markersize=15)
            self.ax.text(x, y + 0.3, opt_action, ha='center', fontsize=12, fontweight='bold', color='blue')
        elif self.state == "GOAL":
            self.ax.text(9.5, 3, "GOAL!", color='green', fontsize=20, ha='center')
        elif self.state == "CRASH":
            self.ax.text(9.5, 3, "CRASHED!", color='red', fontsize=20, ha='center')
        plt.pause(0.01)


# ==========================================================
# 4. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("=== FINE-GRAINED WINDY CHASM (dx=0.05) ===")

    drone_fine = WindyChasmFineMDP(dx=0.05, dy=1, p_center=0.8, p2=0.1, R_goal=100, r_crash=-100)

    print(f"States: {len(drone_fine.get_states())} (Expected ~2660)")
    print("Solving Fine-Grained MDP...")

    drone_fine.value_iteration(theta=1e-4)
    print("Solving Complete.")

    print("Displaying Policy Map (Wide Plot)...")
    visualize_fine_policy_map(drone_fine)
    print("Close Policy Map to start Simulation.")

    env = WindyChasmFineGym(drone_fine)
    state, info = env.reset()
    plt.ion()
    plt.show()

    print("Starting Fine-Grained Simulation...")
    step_count = 0
    terminated = False

    while not terminated:
        opt_str = drone_fine.policy.get(state, 'F')
        gym_act = 0
        if opt_str == 'L':
            gym_act = 1
        elif opt_str == 'R':
            gym_act = 2

        state, reward, terminated, truncated, info = env.step(gym_act)
        step_count += 1
        env.render()

    print(f"Episode finished in {step_count} steps.")
    plt.ioff()
    plt.show()

    while not terminated:
        opt_str = drone_fine.policy.get(state, 'F')
        gym_act = 0
        if opt_str == 'L':
            gym_act = 1
        elif opt_str == 'R':
            gym_act = 2

        state, reward, terminated, truncated, info = env.step(gym_act)
        step_count += 1
        env.render()

    print(f"Episode finished in {step_count} steps.")
    plt.ioff()
    plt.show()