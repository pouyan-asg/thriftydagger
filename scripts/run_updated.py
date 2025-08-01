#!/usr/bin/env python
import torch
import os
import sys
import time
import numpy as np
import gymnasium as gym
import argparse
from thrifty.algos.thriftydagger import thrifty, generate_offline_data
from thrifty.utils.run_utils import setup_logger_kwargs
from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'thrifty')))

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config, refactor_composite_controller_config  # pyright: ignore[reportMissingImports]
# from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper, GymWrapper
from robosuite.devices import Keyboard


class CustomWrapper(GymWrapper):

    def __init__(self, env, render):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self._render = render

    # def reset(self):
    #     """
    #     The action array is [x, y, z, rx, ry, rz, gripper] for pose controllers, 
    #     or [joint1, joint2, ..., joint6, gripper] for joint controllers.
    #     """
    #     r = self.env.reset()
    #     self.render()
    #     settle_action = np.zeros(7)
    #     settle_action[-1] = -1
    #     for _ in range(10):
    #         r = self.env.step(settle_action)
    #         self.render()
    #     self.gripper_closed = False
    #     return r

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = -1
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(settle_action)
            self.render()
        self.gripper_closed = False
        return obs

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        action_[3] = 0.
        action_[4] = 0.
        # self.env.step(action_)
        obs, reward, terminated, truncated, info = self.env.step(action_)
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action[-1]
        for _ in range(10):
            # r1 = self.env.step(settle_action)
            obs, reward, terminated, truncated, info = self.env.step(settle_action)
            if _ == 0:
                print(obs, reward, terminated, truncated, info)
            self.render()
        self.gripper_closed = action[-1] > 0
        # if action[-1] > 0:
        #     self.gripper_closed = True
        # else:
        #     self.gripper_closed = False
        return obs, reward, terminated, truncated, info

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--gen_data', action='store_true', help="True if you want to collect offline human demos")
    parser.add_argument('--iters', type=int, default=5, help="number of DAgger-style iterations")
    parser.add_argument('--targetrate', type=float, default=0.01, help="target context switching rate")
    parser.add_argument('--environment', type=str, default="Lift", help="environment to run")
    # parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--hgdagger', action='store_true')
    # parser.add_argument('--lazydagger', action='store_true')
    parser.add_argument('--eval', type=str, default=None, help="filepath to saved pytorch model to initialize weights")
    # parser.add_argument('--algo_sup', action='store_true', help="use an algorithmic supervisor")
    args = parser.parse_args()
    # render = not args.no_render
    render = True

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    print(f"Logger kwargs: {logger_kwargs}")  # e.g. {'output_dir': '/home/pouyan/phd/imitation_learning/thriftydagger/data/pouyan/pouyan_s0', 'exp_name': 'pouyan'}


    # setup env ...
    # controller_config = load_composite_controller_config(default_controller='OSC_POSE')  # ME
    # controller_config = refactor_composite_controller_config(controller_config='OSC_POSE', robot_type="UR5e", arms='right')  # ME
    # controller_config = load_composite_controller_config(controller="BASIC", robot='UR5e')  # ME

    controller_config = load_composite_controller_config(robot='UR5e')

    config = {
        "env_name": args.environment,
        "robots": "UR5e",
        "controller_configs": controller_config,
    }

    # print(config)

    # Create environment
    if args.environment == 'NutAssembly':
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2, # env has 1 nut instead of 2
            nut_type="round",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    else:
        env = suite.make(
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="agentview",
            # render_camera="birdview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    # env.viewer.set_camera(camera_name=["agentview", "birdview"])
    env = VisualizationWrapper(env, indicator_configs=None)
    env = GymWrapper(env)
    env = CustomWrapper(env, render=render)

    arm_ = 'right'
    config_ = 'single-arm-opposed'
    # input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)
    # if render:
    #     env.viewer.add_keypress_callback("any", input_device.on_press)
    #     env.viewer.add_keyup_callback("any", input_device.on_release)
    #     env.viewer.add_keyrepeat_callback("any", input_device.on_press)
    active_robot = env.robots[arm_ == 'left']


    device = Keyboard(env=env, pos_sensitivity=0.5, rot_sensitivity=3.0)
    env.viewer.add_keypress_callback(device.on_press)
    # env.viewer.add_keyup_callback(device.on_press)
    # env.viewer.add_keyrepeat_callback(device.on_press)

    def hg_dagger_wait():
        # for HG-dagger, repurpose the 'Z' key (action elem 3) for starting/ending interaction
        for _ in range(10):
            # a, _ = device.input2action(
            #     device=input_device,
            #     robot=active_robot,
            #     active_arm=arm_,
            #     env_configuration=config_)
            a = device.input2action()
            env.render()
            time.sleep(0.001)
            if a[3] != 0: # z is pressed
                break
        return (a[3] != 0)

    def expert_pol(o):
        a = np.zeros(7)
        if env.gripper_closed:
            a[-1] = 1.
            # device.grasp = True
            device.grasp_states[device.active_robot][device.active_arm_index] = True
        else:
            a[-1] = -1.
            # device.grasp = False
            device.grasp_states[device.active_robot][device.active_arm_index] = True
        a_ref = a.copy()
        # pause simulation if there is no user input (instead of recording a no-op)
        while np.array_equal(a, a_ref):
            a = device.input2action()
            # a, _ = device.input2action(
            #     device=input_device,
            #     robot=active_robot,
            #     active_arm=arm_,
            #     env_configuration=config_)
            env.render()
            time.sleep(0.001)
        return a

    robosuite_cfg = {'MAX_EP_LEN': 175, 'INPUT_DEVICE': device}
    # if args.algo_sup:
    #     expert_pol = HardcodedPolicy(env).act
    if args.gen_data:
        NUM_BC_EPISODES = 30
        generate_offline_data(env, expert_policy=expert_pol, num_episodes=NUM_BC_EPISODES, seed=args.seed,
                              output_file="robosuite-{}.pkl".format(NUM_BC_EPISODES), robosuite=True, 
                              robosuite_cfg=robosuite_cfg)

    dataset_path = '/home/pouyan/phd/imitation_learning/thriftydagger/data/pouyan/ep_1754065309_9846117'

    if args.hgdagger:
        thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
            seed=args.seed, expert_policy=expert_pol, input_file=dataset_path, robosuite=True, 
            robosuite_cfg=robosuite_cfg, num_nets=1, hg_dagger=hg_dagger_wait, init_model=args.eval)
    # elif args.lazydagger:
    #     lazy(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, noise=0.,
    #      seed=args.seed, expert_policy=expert_pol, input_file="robosuite-30.pkl", robosuite=True, 
    #        robosuite_cfg=robosuite_cfg)
    # else:
    #     thrifty(env, iters=args.iters, logger_kwargs=logger_kwargs, device_idx=args.device, target_rate=args.targetrate, 
    #      	seed=args.seed, expert_policy=expert_pol, input_file="robosuite-30.pkl", robosuite=True, 
    #         robosuite_cfg=robosuite_cfg, q_learning=True, init_model=args.eval)
