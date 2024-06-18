
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(26, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, x, sigma):
        # NeRF-Style positional embedding of x. This helps the function
        # learn high frequency functions more easily.
        # new_x is 26 dimensional.

        x_cos_emb = torch.exp2(torch.linspace(0, 5, 6))
        x_sin_emb = torch.exp2(torch.linspace(0, 5, 6))

        y_cos_emb = torch.exp2(torch.linspace(0, 5, 6))
        y_sin_emb = torch.exp2(torch.linspace(0, 5, 6))

        x_cos_emb = torch.cos(torch.outer(x[:, 0], x_cos_emb) * torch.pi)
        x_sin_emb = torch.sin(torch.outer(x[:, 0], x_sin_emb) * torch.pi)

        y_cos_emb = torch.cos(torch.outer(x[:, 1], y_cos_emb) * torch.pi)
        y_sin_emb = torch.sin(torch.outer(x[:, 1], y_sin_emb) * torch.pi)

        new_x = torch.hstack([x_cos_emb, x_sin_emb, y_cos_emb, y_sin_emb, x])
        return self.net(new_x) / sigma  # Scale network by sigma following NCSNv2


class ScoreMatching():

    def __init__(self):
        self.scorenet = DiffusionModel()

    '''
    Utility function to visualize the score function learned by this model. 
    '''

    def plot_score(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xv, yv = np.meshgrid(x, y)
        arrow_starts = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)

        arrow_ends = (self.scorenet(torch.from_numpy(arrow_starts), 0.03)).detach().numpy()

        plt.quiver(arrow_starts[:, 1], arrow_starts[:, 0], arrow_ends[:, 1], arrow_ends[:, 0])
        plt.title("score function")
        plt.show()

    '''
    Generate initial random noise to denoise.

    A gaussian distribution that roughly fits the ([0,1], [0,1]) range in which the points lie. 
    '''

    def initial_random_samples(self):
        return torch.randn((2000, 2)) / 3 + 0.5

    '''
    Generate geometrically distributed sigmas (sigma_0, sigma_1, ..., sigma_n) starting from smallest going to largest.  
    '''

    def generate_noise_levels(sigma_smallest, sigma_largest, num_levels):
        return torch.tensor(np.exp(np.linspace(np.log(sigma_smallest), np.log(sigma_largest), num_levels))).float()

    '''
    Denoise x_mod using Langevin Dynamics, with:
        - noise_levels being a (L,) dimension tensor containing sigmas
        - iterations specifying the number of iterations to perform
        - lr specifying the learning rate for langevin dynamics 
    '''

    @torch.no_grad()
    def langevin_dynamics_sample(self, x_mod, noise_levels, iterations, lr):

        # TODO: YOUR CODE HERE
        for noise_level in noise_levels:
            x_prev = x_mod.clone()
            alpha = lr * ((noise_level) ** 2 / noise_levels[-1] ** 2)

            for _ in range(iterations):
                z_mu = torch.zeros(2000, 2)
                z_sigma = torch.ones(2000, 2)
                z = torch.normal(z_mu, z_sigma)
                sigma_batch = z_sigma * noise_level
                x_prev = x_prev + (alpha / 2) * scorenet(x_prev, sigma_batch) + torch.sqrt(alpha) * z

            x_mod = x_prev

        return x_mod

    '''
    Calculate denoising score matching loss based on samples. 
        - samples is (2000, 2) dimension tensor containing the dataset
        - noise_levels is a (L) dimension tensor containing your chosen noise levels
    '''

    def denoising_loss(self, samples, noise_levels):

        sigma_chosen = noise_levels[torch.randint(len(noise_levels), (samples.shape[0],))]
        sigma_cat = torch.cat((sigma_chosen.view(2000, 1), sigma_chosen.view(2000, 1)), dim=1)
        sigma_cat_squared = sigma_cat ** 2

        # TODO: YOUR CODE HERE
        x_tilde = torch.normal(samples, sigma_cat)
        loss = self.scorenet(x_tilde, sigma_cat)
        loss += (x_tilde - samples) / sigma_cat_squared
        loss = loss ** 2
        loss = torch.sum(loss)
        return loss

    '''
    Fit self.scorenet to the provided data argument at provided noise_levels. 
    '''

    def train(self, data, noise_levels, learning_rate, iterations):
        optimizer = torch.optim.Adam(self.scorenet.parameters(), lr=learning_rate)

        # TODO: YOUR CODE HERE
        for t in range(iterations):
            optimizer.zero_grad()
            loss = self.denoising_loss(data, noise_levels)
            loss.backward()
            optimizer.step()



def main():
    # TODO: SET HYPERPARAMETERS HERE
    retrain = True
    noise_levels = ScoreMatching.generate_noise_levels(0.1, 0.01, 10)
    training_lr = 0.003
    training_iterations = 2000

    sampling_iterations = 500
    sampling_lr = 0.001

    # Visualize Dataset
    coords = torch.from_numpy(np.load('cs446.npy')).float()
    plt.scatter(coords[:, 1], coords[:, 0])
    plt.title("true samples")
    plt.show()

    # Train or load the model
    sm = None
    if (retrain):
        sm = ScoreMatching()
        sm.train(coords, noise_levels, training_lr, training_iterations)
        torch.save(sm, "model.pth")
    else:
        sm = torch.load("model.pth")

    # Visualize score function
    sm.plot_score()

    # Generate and plot samples
    samples = sm.langevin_dynamics_sample(sm.initial_random_samples(), noise_levels, sampling_iterations,
                                          sampling_lr).numpy()

    plt.scatter(samples[:, 1], samples[:, 0])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("generated samples")
    plt.show()


if __name__ == '__main__':
    main()