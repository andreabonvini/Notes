- ***Describe the differences existing between the Q-Learning and SARSA algorithm***

  SARSA and Q-Learning are two algorithm used to o control using the model free method called temporal difference.

  Q-learning is an example of off-policy learning which means that it learns a target-policy (denoted by $\pi$ ) while following a behavioral policy $\overline{\pi}$. It means that we are, for example, using old policies to learn a new policy, or we are learning a new policy from observing other agents.

  *Q-Learning Update Function*:
  $$
  {Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha \big( \color{red} R_{t+1}+\gamma \max_{a' \in A}  Q(S_{t+1},a') \color{black} - Q(S_t,A_t)\big)}
  $$
   $\pi$ is greedy (there is a max) and $\overline{\pi}$ is $\epsilon$-greedy.

  SARSA is an example of on-policy learning, so it learns the optimal policy based on the actions performed following igit pullts own policy. It samples $A_{t+1}$ from its own policy.

  *SARSA Update Function*:
  $$
  {Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha (\color{red} R_{t+1}+\gamma Q(S_{t+1},A_{t+1}) \color{black} -Q(S_t,A_t))}
  $$
  The two algorithms can perform differently in given situations, for example, in class, we have seen the cliff walking problem. Q-Learning learns an optimal policy along the edge of the cliff because the behavioral policy is $\epsilon$-greedy; while SARSA learnt a safe non-optimal policy away from the edge. This means that if we adopt an $\epsilon$-greedy behavioral policy for Q-Learning, and we use an $\epsilon$-greedy policy for SARSA, we have that:

  - If $\epsilon\neq 0$ SARSA performs better online
  - if $\epsilon\to 0$ gradually both converge to the optimal policy.

