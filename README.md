# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

Initialize the FrozenLake environment, Q-table, and hyperparameters such as γ, α, and epsilon.

Generate episodes using an epsilon-greedy policy, balancing exploration and exploitation.

Calculate the return G from the end of each episode and update the Q-values using the Monte Carlo formula.

Decay epsilon after every episode and repeat the process for 20,000 episodes.

Extract and display the learned greedy policy, Q-table, state values, average reward, and learning curve. 

## Python Program

#### Monte Carlo Control
```
epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate an episode
    episode = generate_episode(epsilon)

    # Calculate total reward
    total_reward = sum(reward for _, _, reward in episode)
    episode_rewards.append(total_reward)

    # Monte Carlo update
    G = 0

    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Incremental update of Q-value
        Q[state, action] += alpha * (G - Q[state, action])

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Optional progress display
    if (episode_num + 1) % 2000 == 0:
        print(
            f"Episode {episode_num + 1}/{num_episodes}, "
            f"Epsilon: {epsilon:.4f}"
        )
```


## Output

```text
Final Q-table:
```
<img width="509" height="400" alt="image" src="https://github.com/user-attachments/assets/83dcaa65-a1e4-43dc-8866-23a106457fb4" />

## Estimated State-Value Function:

<img width="374" height="161" alt="image" src="https://github.com/user-attachments/assets/79c4a099-9279-4711-9d07-a53fad23cf66" />

## Learned Policy:

<img width="458" height="188" alt="Screenshot 2026-08-19 220949" src="https://github.com/user-attachments/assets/b3e30a2c-4d03-48d1-9e22-97e1677d62cd" />

## Average reward over last 1000 episodes: 
<img width="449" height="39" alt="image" src="https://github.com/user-attachments/assets/5fd96a8f-5456-419f-a402-a47fbbfe1606" />



---

## Result
The Q-table is updated through experience collected over 20,000 episodes.

The agent gradually learns the expected return for each state-action pair.

The epsilon-greedy strategy initially encourages exploration and gradually shifts toward exploitation as epsilon decreases from 1.0 to a minimum of 0.05.

The learned greedy policy is obtained using np.argmax(Q, axis=1).

The state-value function is obtained from the maximum Q-value for each state.

The learning curve shows the change in average reward as training progresses. 

---

## Inference

The experiment demonstrates that Monte Carlo Control can learn an optimal or near-optimal policy through repeated interaction with the environment without requiring a model of the environment.

Initially, the agent explores different actions using a high epsilon value. As training continues, epsilon decreases, causing the agent to increasingly select actions with higher learned Q-values. The Q-table is updated using the discounted return G, allowing the agent to estimate the long-term value of each state-action pair.

Therefore, with sufficient episodes, the agent learns a policy that attempts to reach the goal while avoiding the holes in FrozenLake.





---

