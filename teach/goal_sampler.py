"""Temporal variance-driven goal sampling (the core of TEACH).

For each candidate goal we keep a short rolling history of the critic's value
estimate. Goals whose estimates are still moving -- high standard deviation
across that history -- are the ones the policy is actively making progress on,
so they carry the most learning signal and are sampled preferentially.
"""

import random

import numpy as np

import utils

GOAL_POOL_SIZE = 1000


class TemporalVarianceSampler:
    """Maintains the candidate goal pool and its value history.

    Args:
        sample_goals_fun: callable returning `n` goals drawn from the
            environment's goal distribution.
        agent: learner exposing `action_ensemble` and `get_traget_values`.
        val_history: number of past evaluations kept per goal. This is the
            window the variance signal is computed over.
    """

    def __init__(self, sample_goals_fun, agent, val_history=5):
        self.step = 0
        self.all_states = np.asarray(sample_goals_fun(GOAL_POOL_SIZE))
        self.goal_weights = np.zeros((1, GOAL_POOL_SIZE))
        self.val_track = np.ones((val_history, GOAL_POOL_SIZE))

    def build(
        self, sample_goals_fun, agent, n_candidates, strategy, freq, val_history
    ):
        """Build a goal-sampling callable.

        Passing `strategy='uniform'` yields the uniform-sampling
        baseline; any other value selects the temporal-variance curriculum.
        """

        def temporal_variance(values):
            """Shift the value history by one step and score goals by its spread."""
            self.val_track[:-1] = self.val_track[1:]
            self.val_track[-1] = values
            return np.std(self.val_track, axis=0)

        def goal_sampler(obs_dict, action_shape):
            if strategy == 'uniform':
                return sample_goals_fun(1)[0]

            all_states = self.all_states

            # Evaluate every candidate goal from the current observation.
            obs = obs_dict['observation'][np.newaxis, ...]
            input_o = np.repeat(obs, repeats=n_candidates, axis=0)

            dist = agent.action_ensemble(observation=input_o, desired_goal=all_states)
            actions = utils.to_np(dist.mean)

            input_u = np.empty((1, n_candidates, action_shape))
            input_u[0, ...] = actions
            input_o = np.expand_dims(input_o, axis=0)
            g_states = np.expand_dims(all_states, axis=0)

            # Clipped double-Q: average the two target critics.
            val1, val2 = agent.get_traget_values(
                observation=input_o, desired_goal=g_states, action=input_u
            )
            values = np.squeeze(0.5 * (val1 + val2), axis=(0, 2))

            # Refresh the curriculum every `freq` calls, then renormalise.
            if self.step % freq == 0:
                self.goal_weights = temporal_variance(values)
                self.goal_weights /= np.sum(self.goal_weights)
            self.step += 1

            index = random.choices(
                population=np.arange(0, n_candidates),
                weights=self.goal_weights,
                k=1,
            )[0]
            return all_states[index]

        return goal_sampler
