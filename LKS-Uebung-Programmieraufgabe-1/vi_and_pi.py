### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
from lake_envs import *
from frozen_lake import render_single

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters nS, nA, P, gamma are defined as follows:

    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        list of tuples of the form [(probability, nextstate, reward, terminal)] where
            - probability: float
                The probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                Denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                Either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
                True when "nextstate" is a terminal state (hole or goal), False otherwise
    gamma: float
        Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        Defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    neue_value_function = np.zeros(nS)
    while True:
        for state in range(nS):
            for prob, next_state, reward, terminal in P[state][policy[state]]:
                neue_value_function[state] += (reward + gamma * prob * value_function[next_state])
        if max(np.abs(neue_value_function - value_function)) <= tol: break
        value_function = neue_value_function.copy()
        neue_value_function = np.zeros(nS)
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        Defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #
    for state in range(nS):
        actions = []
        for action in range(nA):
            value = sum(
                (reward + gamma * prob * value_from_policy[next_state])
                for prob, next_state, reward, terminal in P[state][action]
            )
            actions.append(value)
        new_policy[state] = np.argmax(actions)
    ############################

    return new_policy

def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        Defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)


    ############################
    # YOUR IMPLEMENTATION HERE #
    for _ in range(10):
        value_from_policy = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_from_policy, gamma)
        if (new_policy == policy).all(): break
        else: policy = new_policy
    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        Defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    count = 0
    neue_value_function = np.zeros(nS)
    while True:
        count += 1
        for state in range(nS):
            for prob, next_state, reward, terminal in P[state][policy[state]]:
                neue_value_function[state] += (reward + gamma * prob * value_function[next_state])

        policy = policy_improvement(P, nS, nA, value_function, gamma)
        if max(np.abs(neue_value_function - value_function)) <= tol and count>10: break
        value_function = neue_value_function.copy()
        neue_value_function = np.zeros(nS)
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
   # env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
