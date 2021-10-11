import src.temporal_playground_env.playgroundnavv1

from gym.envs.registration import register

for v in ['1']:
    register(id='TemporalPlayground-v' + v,
             entry_point='src.temporal_playground_env.playgroundnavv' + v + ':TemporalPlaygroundV' + v,
             max_episode_steps=50)

    register(id='TemporalPlaygroundHuman-v' + v,
             entry_point='src.temporal_playground_env.playgroundnavv' + v + ':TemporalPlaygroundV' + v,
             max_episode_steps=50,
             kwargs=dict(human=True, render_mode=True))

    register(id='TemporalPlaygroundRender-v' + v,
             entry_point='src.temporal_playground_env.playgroundnavv' + v + ':TemporalPlaygroundV' + v,
             max_episode_steps=50,
             kwargs=dict(human=False, render_mode=True))
