from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

unity_env = UnityEnvironment(file_name="/env/3dBallCustom-v0.x86_64", worker_id=1)
env = UnityToGymWrapper(unity_env, 0, uint8_visual=True)

state_ = env.reset()

for i in range(100):
    a = env.step(env.action_space.sample())
    env.render()
