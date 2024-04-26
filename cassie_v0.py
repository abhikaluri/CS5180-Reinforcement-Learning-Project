""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import gym
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0

class CassieEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)

        self.keyframe_qpos =  " 0 0 1.0059301 1 0 0 0 0.00449956 0 0.497301 0.97861 -0.0164104 0.0177766  -0.204298 -1.1997 0 1.42671 -2.25907e-06 -1.52439 1.50645 -1.59681 -0.00449956 0 0.497301 0.97874 0.0038687 -0.0151572 -0.204509 -1.1997 0 1.42671 0 -1.52439 1.50645 -1.59681 "
                                
                                
                            
    def _setup(self,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq
            reset_type = "init",            # none; init; random
            target_type = "generate",       # generate; switch; fixed
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pose_thd = 0.35,
            weight_bodyname = None,
            weight_range = None,
            **kwargs,
        ):
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value

        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=viz_site_targets,
                **kwargs,
                )


    def get_termination(self):
        # Extract quaternion for the pelvis
        orientation_quat = self.sim.data.body('cassie-pelvis').xquat.copy()  # Correct body name
        #orientation_quat = train_env.unwrapped.sim.data.body_xquat('cassie-pelvis').copy()

        roll, pitch, yaw = self.get_intrinsic_euler(orientation_quat)
        
        # Check height condition
        height = self.sim.data.joint('cassie_z').qpos[0].copy()  # Assuming index 2 is height
        height_condition = not (-0.2 < height < 2.0) # "upright" cassie starts at height=0
        
        # Check tilt conditions based on roll and pitch angles
        tilt_condition = abs(pitch) > np.deg2rad(30) or abs(roll) > np.deg2rad(30)  # Thresholds in radians
        
        # Determine if episode should end
        if height_condition or tilt_condition:
            termination_reason = 'fallen' if height_condition else 'tilted'
            return True, {'termination_reason': termination_reason}
        
        return False, {}
    



    def get_intrinsic_euler(self, quat):
        """Convert quaternion to intrinsic euler angles XYZ (pitch, roll, yaw)"""
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return roll, pitch, yaw

    def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert Euler Angles to Quaternion.
        roll, pitch, yaw : Rotation around x, y and z axes (in radians)
        Returns a quaternion in the form [w, x, y, z]
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['qpos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        joint_list = ['left-foot', 'left-hip-pitch', 'left-hip-roll', 'left-hip-yaw', 'left-knee',
                      'right-foot', 'right-hip-pitch', 'right-hip-roll', 'right-hip-yaw', 'right-knee']
        qpos_list = np.zeros(len(joint_list))
        qvel_list = np.zeros(len(joint_list))

        for jnt_idx, jnt_name in enumerate(joint_list):
            joint_id = sim.model.joint_name2id(jnt_name)
            qpos_list[jnt_idx] = sim.data.qpos[joint_id]
            qvel_list[jnt_idx] = sim.data.qvel[joint_id]

        obs_dict['qpos'] = qpos_list
        obs_dict['qvel'] = qvel_list * self.dt

        if not hasattr(self, 'target_jnt_value') or self.target_jnt_value is None:
            self.target_jnt_value = np.zeros(len(joint_list))

        if isinstance(self.target_jnt_value, (int, float)):
            self.target_jnt_value = np.zeros(len(joint_list))

        if len(self.target_jnt_value) < len(joint_list):
            padded_value = np.zeros(len(joint_list))
            padded_value[:len(self.target_jnt_value)] = self.target_jnt_value
            self.target_jnt_value = padded_value

        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na > 0 else np.zeros_like(qpos_list)
        obs_dict['pose_err'] = self.target_jnt_value - qpos_list

        return obs_dict

    def get_cassie_obs_dict(self):
        obs_dict = {}
        #obs_dict['time'] = np.array([self.sim.data.time])

        joint_list = ['left-foot', 'left-hip-pitch', 'left-hip-roll', 'left-hip-yaw', 'left-knee',
                      'right-foot', 'right-hip-pitch', 'right-hip-roll', 'right-hip-yaw', 'right-knee']
        qpos_list = np.zeros(len(joint_list))
        qvel_list = np.zeros(len(joint_list))

        for jnt_idx, jnt_name in enumerate(joint_list):
            joint_id = self.sim.model.joint_name2id(jnt_name)
            qpos_list[jnt_idx] = self.sim.data.qpos[joint_id]
            qvel_list[jnt_idx] = self.sim.data.qvel[joint_id]

        obs_dict['qpos'] = qpos_list
        obs_dict['qvel'] = qvel_list * self.dt

        return obs_dict
    

    def calculate_standing_reward(self, obs_dict):
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']

        # Constants for reward calculation
        ideal_height = 0.75  # meters
        height_tolerance = 0.1  # meters
        velocity_smoothing = 0.1  # small constant to smooth velocity
        orientation_smoothing = 0.05  # for orientation reward calculation

        # Calculate center of mass height and velocity
        com_height = -0.5 - self.sim.data.joint('cassie_z').qpos[0].copy()
        com_vel = self.sim.data.joint('cassie_x').qvel[0].copy()

        # Orientation penalties
        orientation_quat = self.sim.data.body('cassie-pelvis').xquat.copy()
        roll, pitch, yaw = self.get_intrinsic_euler(orientation_quat)
        orientation_penalty = 1 - np.cos(roll) * np.cos(pitch)  # less steep penalty

        # Compute height reward using a Gaussian function
        height_deviation = np.abs(com_height - ideal_height)
        reward_height = np.exp(-np.square((height_deviation - height_tolerance) / orientation_smoothing))

        # Compute velocity reward using smoothing
        smoothed_com_vel = np.sqrt(com_vel**2 + velocity_smoothing)
        reward_vel = np.exp(-smoothed_com_vel)

        # Energy expenditure (assuming qvel relates to joint velocities)
        energy_expenditure = np.sum(np.square(qvel))
        reward_energy = np.exp(-energy_expenditure)

        # Determine if the episode should end
        termination_result, termination_info = self.get_termination()
        if termination_result:
            if termination_info.get('termination_reason') in ['fallen', 'tilted']:
                return 0  # Zero reward for falling or tilting

        # Combine rewards with weights
        weights = {'height': 0.4, 'velocity': 0.2, 'orientation': 0.2, 'energy': 0.2}
        total_reward = (weights['height'] * reward_height +
                        weights['velocity'] * reward_vel +
                        weights['orientation'] * orientation_penalty +
                        weights['energy'] * reward_energy)

        return total_reward



    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)
        if self.sim.model.na !=0: act_mag= act_mag/self.sim.model.na
        far_th = 4*np.pi/2

        standing_reward = self.calculate_standing_reward(obs_dict) 
        
         
        rwd_dict = collections.OrderedDict((
            ('pose',    -1.*pose_dist),
            ('bonus',   1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd)),
            ('penalty', -1.*(pose_dist>far_th)),
            ('act_reg', -1.*act_mag),
            ('standing_reward', standing_reward),  # Add the custom standing reward
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<self.pose_thd),
            ('done',    pose_dist>far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_cassie_rew(self, obs_dict):
        standing_reward = self.calculate_standing_reward(obs_dict)
        return standing_reward

    # generate a valid target pose
    def get_target_pose(self):
        if self.target_type == "fixed":
            return self.target_jnt_value
        elif self.target_type == "generate":
            return self.np_random.uniform(low=self.target_jnt_range[:,0], high=self.target_jnt_range[:,1])
        else:
            raise TypeError("Unknown Target type: {}".format(self.target_type))

    # update sim with a new target pose
    def update_target(self, restore_sim=False):
        if restore_sim:
            qpos = self.sim.data.qpos[:].copy()
            qvel = self.sim.data.qvel[:].copy()
        # generate targets
        self.target_jnt_value = self.get_target_pose()
        # update finger-tip target viz
        if self.target_type == 'fixed':
            self.sim.data.qpos[:] = self.target_jnt_value
        else:
            self.sim.data.qpos[:] = self.target_jnt_value.copy()
        self.sim.forward()
        for isite in range(len(self.tip_sids)):
            self.sim.model.site_pos[self.target_sids[isite]] = self.sim.data.site_xpos[self.tip_sids[isite]].copy()
        if restore_sim:
            self.sim.data.qpos[:] = qpos[:]
            self.sim.data.qvel[:] = qvel[:]
        self.sim.forward()        

    def get_cassie_vec(self, obs_dict):
        obsvec = np.zeros(0)
        for key in obs_dict.keys():
            obsvec = np.concatenate([obsvec, obs_dict[key].ravel()])
        return np.array(obsvec, dtype=np.float32)

    # reset_type = none; init; random
    # target_type = generate; switch
    def reset(self, init_cassie=None):
        
        obs = super().reset(reset_qpos=self.sim.model.keyframe('stand').qpos.copy())
        #obs = # Convert your obs dict to a vector

        obsvec = self.get_cassie_vec(self.get_cassie_obs_dict())
        #print(obs_dict)
        return np.array(obsvec, dtype=np.float32)