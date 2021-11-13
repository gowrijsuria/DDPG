import numpy as np

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, action_dim=1):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu