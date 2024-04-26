###############################################################Template for SB3 ##################################################################
import gymnasium
import gym
import numpy as np
from gymnasium import spaces

#from myosuite.envs.myo.myobase.cassie_v0 import CassieEnvV0

class CassieWrapper(gymnasium.Env):  # previously it was just Env 
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(20,), dtype=np.float32)

        # Start your Cassie env here
        self.cassie_env = gym.make('CassieWalk-v0') # Only for Myosuite (gym)    ## i think we need to add self here 
    
    def step(self, action):
        
        self.cassie_env.step(action)

        obs_dict = self.cassie_env.get_cassie_obs_dict()

        observation = self.cassie_env.get_cassie_vec(obs_dict)
        reward = self.cassie_env.get_cassie_rew(obs_dict)
        info = {}
        
        # Check for termination conditions
        # 1 - Body height
        # 2 - Body tilt (pitch values) - quaternion to intrinsic euler (XYZ)
        # 3 - Body roll

        # For 2 and 3, env.sim.data.body('cassie-pelvis').xquat.copy()
        # write conversion function
        # get_intrinsicEuler(quat):
            # w, x, y, z = quat
            # ...

        truncated,_ = self.cassie_env.get_termination() # return boolean
        terminated,_ = self.cassie_env.get_termination()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        #qpos = self.cassie_env.sim.model.keyframe('stand').qpos.copy()
        observation = self.cassie_env.reset()
        #print(f"type: {type(observation)}, {observation}")
        info = {}
        return observation, info

    def render(self, mode='human'):
        return self.cassie_env.render(mode=mode)

    def close(self):
        self.cassie_env.close()
