"""Optional registration of the particle-maze navigation tasks.

The maze environment itself is *not* included in this repository: it is the
`ParticleMazeEnv` from the VDS codebase (https://github.com/zzyunzhi/vds,
`baselines/envs/maze/`). To run the maze tasks, copy `maze.py` and
`maze_layouts.py` from there into this directory.

The manipulation tasks need none of this and work out of the box.
"""

from gymnasium.envs.registration import register

for grid_name in ['a', 'b', 'c']:
    register(
        id=f'Maze{grid_name.capitalize()}-v0',
        entry_point='maze:ParticleMazeEnv',
        kwargs={'grid_name': grid_name, 'reward_type': 'sparse'},
        max_episode_steps=50,
    )
