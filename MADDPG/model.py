# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
class critic_NN(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_NN, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class acot_NN(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(acot_NN, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        # Action space
        self.action_space = [gym.spaces.Box(low=-5.0, high=10.0, shape=(1,), dtype=float),
                             gym.spaces.Box(low=-15.0, high=15.0, shape=(1,), dtype=float)]
        self.amplitude = torch.tensor([(self.action_space[0].high - self.action_space[0].low) / 2.0, (self.action_space[1].high - self.action_space[1].low) / 2.0], device=args.device ,dtype=torch.float32)
        self.mean = torch.tensor([(self.action_space[0].high + self.action_space[0].low) / 2.0, (self.action_space[1].high + self.action_space[1].low) / 2.0], device=args.device, dtype=torch.float32)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        # gain = nn.init.calculate_gain('leaky_relu')
        # gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        model_out = self.tanh(model_out)
        noise = torch.rand_like(model_out)
        
        # if model_out.dim() == 1:
        #     # 배치 크기가 1인 경우
        #     action1 = (torch.tanh(model_out[0]) * self.amplitude[0]) + self.mean[0]
        #     action2 = (torch.tanh(model_out[1]) * self.amplitude[1]) + self.mean[1]
        #     policy = torch.cat((action1.unsqueeze(1), action2.unsqueeze(1)), dim=1).squeeze()
        # else:
        #     # 다른 배치 크기인 경우
        #     action1 = (torch.tanh(model_out[:, 0]) * self.amplitude[0]) + self.mean[0]
        #     action2 = (torch.tanh(model_out[:, 1]) * self.amplitude[1]) + self.mean[1]
        #     policy = torch.cat((action1.unsqueeze(1), action2.unsqueeze(1)), dim=1)
        
        # In case, model_out is in the range of [-1, 1]. There will be noise [-0.02, 0.02]
        policy = model_out + (2 * noise - 1) * 0.1
        
        if model_original_out == True:
            return model_out, policy  # model_out criterion을 위한 반환
        return policy
