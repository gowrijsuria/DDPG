import torch
import torch.nn as nn

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import random
from collections import namedtuple, deque

from noise import OrnsteinUhlenbeckActionNoise as OUNoise

Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward'))

class Trainer:
    def __init__(self, args, env, memory, ActorNet, ActorTargetNet, Actor_optimizer, Actor_scheduler, 
                        CriticNet, CriticTargetNet, Critic_optimizer, Critic_scheduler, device):
        super().__init__()
        self.args = args
        self.env = env
        self.memory = memory
        self.ActorNet = ActorNet
        self.ActorTargetNet = ActorTargetNet
        self.Actor_optimizer = Actor_optimizer
        self.Actor_scheduler = Actor_scheduler
        self.CriticNet = CriticNet
        self.CriticTargetNet = CriticTargetNet
        self.Critic_optimizer = Critic_optimizer
        self.Critic_scheduler = Critic_scheduler
        self.OUNoise = OUNoise()
        self.device = device
        self.steps_done = 0
        random.seed(self.args.seed_val)
        self.save_dir = os.path.join(self.args.save_path, self.args.experiment, '')

    def save_state(self, net, save_path):
        print('==> Saving models ...')
        state = {
                'net_state_dict': net.state_dict()
                }
        dir_name = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(dir_name):
            print("Creating Directory: ", dir_name)
            os.makedirs(dir_name)
        torch.save(state, str(save_path) + '.pth')

    def savePlots(self, actor_losses, critic_losses, rewardVsEpisode):
        plt.figure()
        plt.plot(actor_losses)
        plt.title("Actor Loss vs Step")
        plt.savefig(self.save_dir+ 'ActorLossVsStep.png')

        plt.figure()
        plt.plot(critic_losses)
        plt.title("Critic Loss vs Step")
        plt.savefig(self.save_dir+ 'CriticLossVsStep.png')

        plt.figure()
        plt.plot(rewardVsEpisode)
        plt.title("Reward vs Episode")
        plt.savefig(self.save_dir+'RewardVsEpisode.png')

    def Images_to_Video(self, recordingFrames):
        vid_path = self.save_dir + 'video.mp4'
        frame_rate = 1
        h,w,c = recordingFrames[0].shape
        frame_size = (w,h)
        out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

        imgs_path = os.path.join(self.save_dir, 'imgs')
        for i in range(len(recordingFrames)):
            recordingFrame = cv2.cvtColor(recordingFrames[i], cv2.COLOR_RGB2BGR)
            img_final = os.path.join(imgs_path,  str(i) + '.png')
            self.check_path(img_final)
            cv2.imwrite(img_final, recordingFrame)
            out.write(recordingFrame)
        out.release() 

    def check_path(self, fname):
        dir_name = '/'.join(fname.split('/')[:-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def getBatch(self, curr_state):
        image_shape = curr_state['image'].shape
        obstacles_shape = curr_state['area_obstacle'].shape
        images_batch = torch.zeros((self.args.batch_size, image_shape[1], image_shape[2], image_shape[3]), device=self.device) 
        obstacles_batch = torch.zeros((self.args.batch_size, obstacles_shape[1]), device=self.device)

        if len(self.memory) + 1 < self.args.batch_size:
            image_zero = torch.zeros_like(curr_state['image'])
            area_zero = torch.zeros_like(curr_state['area_obstacle'])
            remaining_size = self.args.batch_size - len(self.memory) - 1
            for i in range(remaining_size):
                images_batch = torch.cat((images_batch, image_zero), dim=0)
                obstacles_batch = torch.cat((obstacles_batch, area_zero), dim=0)

            print("len memory:", len(self.memory))
            print("memory:", self.memory)
            for i in range(len(self.memory)):
                images_batch = torch.cat((images_batch, self.memory.get(i)[0]['image']), dim=0)
                obstacles_batch = torch.cat((obstacles_batch, self.memory.get(i)[0]['area_obstacle']), dim=0)
        else:
            sampled_batch = self.memory.sampleBatch(self.args.batch_size - 1)
            for i in range(self.args.batch_size - 1):
                images_batch = torch.cat((images_batch, sampled_batch[i][0]['image']), dim=0)
                obstacles_batch = torch.cat((obstacles_batch, sampled_batch[i][0]['area_obstacle']), dim=0)

        states_batch = {'image': torch.cat([curr_state['image'], images_batch], dim = 0), 
                'area_obstacle': torch.cat([curr_state['area_obstacle'], obstacles_batch], dim = 0)} 
        return states_batch

    def select_action(self, curr_state, epsilon_thres):
        states_batch = self.getBatch(curr_state)

        with torch.no_grad():
            action = self.ActorNet(states_batch['image'], states_batch['area_obstacle'])
        action = action.detach().cpu().numpy().squeeze()

        ou_noise = self.OUNoise.noise()
        random_num = random.random()
        if random_num > epsilon_thres:
            # choose action via exploitation
            action = torch.tensor([action[0]])
        else:
            # choose action via exploration
            action = torch.tensor([np.clip(action[0] + ou_noise[0], -1, 1)])
        
        return action

    def optimisePolicyNet(self):
        if len(self.memory) < self.args.batch_size:
            return 
        sampled_batch = self.memory.sampleBatch(self.args.batch_size)

        states = {'image': torch.stack([each[0]['image'] for each in sampled_batch], dim=1).squeeze(0).to(self.device), 
                'area_obstacle': torch.stack([each[0]['area_obstacle'] for each in sampled_batch], dim=1).squeeze(0).to(self.device)}
        actions = torch.tensor([each[1] for each in sampled_batch]).to(self.device).unsqueeze(-1)
        next_states = {'image': torch.stack([each[2]['image'] for each in sampled_batch], dim=1).squeeze(0).to(self.device), 
                'area_obstacle': torch.stack([each[2]['area_obstacle'] for each in sampled_batch], dim=1).squeeze(0).to(self.device)}
        rewards = torch.tensor([each[3] for each in sampled_batch]).to(self.device)

        Q_value = self.CriticNet(states['image'], states['area_obstacle'], actions.float())
        next_Q_value = torch.zeros(self.args.batch_size, device=self.device)
        with torch.no_grad():
            next_action_value = self.ActorTargetNet(next_states['image'], next_states['area_obstacle'])
            next_Q_value = self.CriticTargetNet(next_states['image'], next_states['area_obstacle'], next_action_value.float()).squeeze(-1)
            next_action_value = next_action_value.detach()
            next_Q_value = next_Q_value.detach()

        episode_ends = []
        for i in range(next_states['image'].shape[0]):
            if torch.all(torch.eq(next_states['image'][i], torch.zeros(states['image'][0].shape).to(self.device))):
                episode_ends.append(i)
        episode_ends = torch.tensor(episode_ends)

        for i in episode_ends:
            next_Q_value[i] = torch.tensor(0.0)
        expected_state_action_values = (next_Q_value * self.args.gamma) + rewards

        criterion = nn.MSELoss()
        critic_loss = criterion(Q_value, expected_state_action_values.unsqueeze(1))

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()
        self.Critic_scheduler.step()

        action_value = self.ActorNet(states['image'], states['area_obstacle'])
        Q_value_policy = self.CriticNet(states['image'], states['area_obstacle'], action_value).squeeze(-1)

        actor_loss = -(Q_value_policy).mean()
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()
        self.Actor_scheduler.step()

        self.soft_update(self.ActorTargetNet, self.ActorNet, self.args.tau)
        self.soft_update(self.CriticTargetNet, self.CriticNet, self.args.tau)
        return actor_loss, critic_loss

    def fillReplayMemory(self, replay_memory_size):
        for i in range(replay_memory_size):
            state, rgbImage = self.env.ImageAtCurrentPosition()
            epsilon_thres = 1.0
            action = self.select_action(state, epsilon_thres)
            done, next_state, reward, rgbImage = self.env.step(action)
            self.memory.push((state, action, next_state, reward))
            if done or i % self.args.max_steps == 0:
                self.env.reset()
            else:
                state = next_state

        self.env.reset()

    def train(self):
        save_outputfile = self.save_dir + 'output.txt'
        self.check_path(save_outputfile)
        f = open(save_outputfile, 'w')
        actor_losses = []
        critic_losses = []
        rewardVsEpisode = []

        self.fillReplayMemory(self.args.replay_memory_size)
        for episode in range(1, self.args.num_episodes+1):
            total_reward = 0
            state, rgbImage = self.env.ImageAtCurrentPosition()
            recordingFrames = []
            recordingFrames.append(rgbImage)
            step = 0
            while True:
                self.steps_done+=1 # for epsilon_decay
                epsilon_thres = self.args.epsilon_stop + (self.args.epsilon_start - self.args.epsilon_stop)* \
                            np.exp(-1. * self.steps_done/self.args.epsilon_decay) 
                action = self.select_action(state, epsilon_thres)
                done, next_state, reward, rgbImage = self.env.step(action)
                total_reward += reward
                self.memory.push((state, action, next_state, reward))
                recordingFrames.append(rgbImage)

                actor_loss, critic_loss = self.optimisePolicyNet()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                if done or step == self.args.max_steps:
                    print('Episode: {}'.format(episode),'Total reward: {}'.format(total_reward[0][0].cpu().numpy()))

                    rewardVsEpisode.append(total_reward)
                    averageReward = torch.mean(torch.FloatTensor(rewardVsEpisode[episode-min(episode,self.args.reward_average_window):episode+1]))
                    f.write('Episode: {}, Total reward: {} \n'.format(episode, total_reward[0][0].cpu().numpy()))
                    self.env.reset()
                    break
                else:
                    state = next_state
                step+=1
            if averageReward >= self.args.averageRewardThreshold or episode == self.args.num_episodes:
                print("rewardVsEpisode")
                print(rewardVsEpisode)
                self.Images_to_Video(recordingFrames)
                self.savePlots(actor_losses, critic_losses, rewardVsEpisode)
                self.save_state(self.ActorNet, self.save_dir + self.args.experiment)
                self.save_state(self.CriticNet, self.save_dir + self.args.experiment)
                break

        
