import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from algo.MARL_wrapper import update_trainers
from algo.replay_buffer import ReplayBuffer
from algo.model import acot_NN, critic_NN

from envs.level0.tmps_env import env_level0

import wandb

# For logging
wandb.init(
    # set the wandb project where this run will be logged
    project="ai-crew",
    # track hyperparameters and run metadata
    config={
    "obs_format": "relative_pos",
    "architecture": "MLP",
    "dataset": "accelerated_env",
    "epochs": "None",
    }
    )

def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    # input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True: # For continual learning
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name+'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name+'t_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(env.n):
        actors_cur[i] = acot_NN(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = critic_NN(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = acot_NN(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = critic_NN(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """ 
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
        (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...'+' '*100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, _, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx) # Note_The func is not the same as others
                
            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float) # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device) # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1) # q 
            q_ = critic_t(obs_n_n, action_tar).reshape(-1) # q_ 
            tar_value = q_*arglist.gamma*done_n + rew # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new 
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-1*loss_pse+loss_a).backward()  # 작을수록 좋다
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                arglist.scenario_name, time_now, game_step))
            if not os.path.exists(model_file_dir): # make the path
                os.makedirs(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao) 
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao) 

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def train(arglist, video):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """

    env_config = {
    "num_agents": 4,
    "obs_box_size": 50,
    "init_pos": ((60., 110.), (200., 140.), (60., 240.), (210., 220.)),
    "dynamic_delta_t": 0.01
    }

    env = env_level0(env_config)

    # Action space
    action_space = env.agents[0].action_space
    action_bound = [np.array([action_space[0].low[0], action_space[1].low[0]]), np.array([action_space[0].high[0], action_space[1].high[0]])]
    
    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [2,2,2,2]  # e.g [8, 10, 10] , currently [4,4,4,4] -> [2,2,2,2] relative position
    action_shape_n = [2,2,2,2] # no need for stop bit # e.g [5, 5, 5]
    num_adversaries = None # no need for current trainer
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)
    
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_cnt = 0
    update_cnt = 0
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
        
    # obs_size [(0, 2), (2, 4), (4, 6), (6, 8)]
    # action_size [(0, 2), (2, 4), (4, 6), (6, 8)]

    print('=3 starting iterations ...')
    print('=============================')

    reset_arg = {
        'episode': 0
    }
    obs_n = env.reset(**reset_arg)
    # 4 agents reset
    obs_n = np.array([[0, 0] for _ in range(4)])  # pjhae
    
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data

        print('steps gone:{} episode gone:{}'.format(game_step, episode_gone))

        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() * (action_bound[1] - action_bound[0])/2.0 + (action_bound[1] + action_bound[0])/2.0 \
                for agent, obs in zip(actors_cur, obs_n)]
            
            # pjhae
            _action_n = [] 
            for _, action in enumerate(action_n):
                _action_n.append((np.array([action[0]]), np.array([action[1]]), 7, 0, 0))   # Can check at "agent_base.py > self.action_space"
            _action_n = tuple(_action_n)

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(_action_n)

            # pjhae
            # new_obs_n = np.concatenate((info_n['objects_pos'][0:4, :], info_n['objects_pos'][4:8, :]), axis=1) # Global goal pos 4*4
            agent_pos = info_n['objects_pos'][0:4, :]
            goal_pos  = info_n['objects_pos'][4:8, :]

            new_obs_n = goal_pos - agent_pos

            rew_n = -np.linalg.norm(new_obs_n, axis=1) # 크기가 4인 벡터로 아웃폿 [-1, -2, -3, -1]

            done_n = np.array([False for _ in range(len(rew_n))])
            done_n[-rew_n < 2] = True
            
            if np.all(done_n) == True:
                print("goal in")
            
            if (episode_cnt >= arglist.per_episode_max_len-1 or np.all(done_n) == True):
                done_n = [True for _ in range(len(rew_n))]

            # save the experience
            memory.add(obs_n, np.concatenate(action_n), rew_n , new_obs_n, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew
            
            # train our agents 
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(\
                arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            terminal = (episode_cnt >= arglist.per_episode_max_len-1)
            if np.all(done_n) or terminal:
                obs_n =  env.reset(**reset_arg)
                obs_n = np.array([[0,0] for _ in range(4)])  # pjhae
                agent_info.append([[]])
                episode_rewards.append(0)
                for a_r in agent_rewards:   
                    a_r.append(0)
                continue

        # 비디오 추가, reward 바꾸기, 초기위치 쉽게
        # evalution
        if episode_gone % 20 == 0 :
            video.init(enabled=True)
            for _ in range(5):
                obs_n =  env.reset(**reset_arg)
                obs_n = np.array([[0,0] for _ in range(4)])  # pjhae
                
                #TODO I think action_target output should be used, and model_original_out=True
                action_n_test = []
                for episode_cnt in range(arglist.per_episode_max_len):
                    for actor, obs in zip(actors_tar, obs_n):
                        model_out= actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)[0].detach().cpu().numpy() * (action_bound[1] - action_bound[0])/2.0 + (action_bound[1] + action_bound[0])/2.0
                        action_n_test.append(model_out)
                    # pjhae
                    _action_n_test = [] 
                    for i, action in enumerate(action_n_test):
                        _action_n_test.append((np.array([action[0]]), np.array([action[1]]), 7, 0, 0))
                    _action_n_test = tuple(_action_n_test)

                    new_obs_n, rew_n, done_n, info_n = env.step(_action_n_test)

                    video.record(env.render(mode='rgb_array'))

                    agent_pos = info_n['objects_pos'][0:4, :]
                    goal_pos  = info_n['objects_pos'][4:8, :]

                    new_obs_n = goal_pos - agent_pos

                    rew_n = -np.linalg.norm(new_obs_n, axis=1)

                        
                    obs_n = new_obs_n
                    
                    terminal = (episode_cnt >= arglist.per_episode_max_len-1)
                    if np.all(done_n) or terminal:
                        break

            print("evaluation is finished at", episode_gone, "th episode" )
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
            print('episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r))

            wandb.log({"mean_agents_r": np.sum(mean_agents_r)})

            video.save('test_{}.mp4'.format(episode_gone))
            video.init(enabled=False)
