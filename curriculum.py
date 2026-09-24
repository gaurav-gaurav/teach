"""Wiring between an environment's goal space and the TEACH sampler."""

import gym

from goal_env import CurriculumGoalEnv
from goal_sampler import TemporalVarianceSampler


def make_env(env_id, seed=1):
    """Create `env_id` wrapped so the curriculum can set its goal."""
    return CurriculumGoalEnv(gym.make(env_id), seed=seed)


def build_goal_samplers(env_id, agent, n_candidates, freq, val_history):
    """Return `(curriculum_sampler, uniform_sampler)` for `env_id`.

    The uniform sampler is the baseline the paper compares against, and is what
    the evaluation environment uses so evaluation goals stay unbiased.
    """
    probe = make_env(env_id)
    probe.unwrapped.goal = probe.unwrapped._sample_goal()
    probe.reset(reset_goal=False)

    def sample_goals(size):
        return [probe.unwrapped._sample_goal() for _ in range(size)]

    sampler = TemporalVarianceSampler(sample_goals, agent, val_history=val_history)

    curriculum = sampler.build(
        sample_goals, agent, n_candidates=n_candidates, strategy='variance',
        freq=freq, val_history=val_history,
    )
    uniform = sampler.build(
        sample_goals, agent, n_candidates=1, strategy='uniform',
        freq=freq, val_history=val_history,
    )
    return curriculum, uniform
