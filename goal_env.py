"""Environment wrapper that lets a curriculum choose the episode goal."""

from gym import Wrapper


class CurriculumGoalEnv(Wrapper):
    """Delegates goal selection on reset to an injected sampler.

    Call `set_goal_sampler` before the first `reset`. With no sampler set, or
    `reset(reset_goal=False)`, the environment's own goal distribution is used.
    """

    def __init__(self, env, seed):
        super().__init__(env)
        self._goal_sampler = None
        self._is_maze = 'Maze' in env.spec.id
        if self._is_maze:
            env.unwrapped.reset(reset_goal=False)
            env.unwrapped.seed(seed=seed)
        else:
            env.reset(seed=seed)

    def set_goal_sampler(self, sampler):
        self._goal_sampler = sampler

    def reset(self, reset_goal=True):
        # Reset the step counter directly: gym.wrappers.TimeLimit is bypassed
        # because we reset the simulator without going through Env.reset.
        self.env._elapsed_steps = 0

        reset_ok = False
        while not reset_ok:
            if self._is_maze:
                reset_ok = self.unwrapped.reset(reset_goal=False)
            else:
                reset_ok = self.unwrapped._reset_sim()

        if reset_goal and self._goal_sampler is not None:
            obs = self.unwrapped._get_obs()
            self.unwrapped.goal = self._goal_sampler(
                obs_dict=obs, action_shape=self.action_space.shape[0]
            )
        else:
            self.unwrapped.goal = self.unwrapped._sample_goal().copy()

        return self.unwrapped._get_obs()

    def envs_op(self, op_name, **kwargs):
        return getattr(self.env, op_name)(**kwargs)
