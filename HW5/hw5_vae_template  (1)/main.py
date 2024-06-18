def main():
    # read the function arguments
    args = parse_args()

    # set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # train the model
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()

    # Generate digits using the VAE
    def visualize_data_space(self):
        # TODO: Sample 10 z from prior
        mu_0 = torch.zeros(10, self.latent_dimension)
        sigma_1 = torch.ones(10, self.latent_dimension)
        z = self.sample_diagonal_gaussian(mu_0, sigma_1)

        # TODO: For each z, plot p(x|z)
        figure = np.zeros((20, 28, 28))
        distribution_arr = torch.zeros(10, 784)
        for i in range(10):
            distribution = self.decoder(z[i])
            # store the distribution of each x for sampling
            distribution_arr[i] = distribution
            distribution = array_to_image(distribution.detach().numpy())
            # store the image to the final figure
            figure[i] = distribution

        # TODO: Sample x from p(x|z)
        for i in range(10):
            x_sample = self.sample_Bernoulli(distribution_arr[i])
            x_sample = array_to_image(x_sample.detach().numpy())
            # store the image to the final figure
            figure[i + 10] = x_sample

        # TODO: Concatenate plots into a figure (use the function concat_images)
        final_image = concat_images(figure, 10, 2, padding=3)

        # TODO: Save the generated figure and include it in your report
        plt.imshow(final_image)
        plt.show()

    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot
    # by the class label for the input data. Each point in the plot is colored by the class label for
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        # TODO: Encode the training data self.train_images
        mu, sigma = self.encoder(self.train_images)

        # TODO: Take the mean vector of each encoding
        mu_x = mu.detach().numpy()[:, 0:1]
        mu_y = mu.detach().numpy()[:, 1:2]

        # TODO: Plot these mean vectors in the latent space with a scatter
        # Colour each point depending on the class label
        color_dict = {0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple", 5: "brown", 6: "pink", 7: "gray",
                      8: "olive", 9: "cyan"}
        for i in range(mu.shape[0]):
            # get the class label for each sample
            label = torch.argmax(self.train_labels[i]).item()
            plt.scatter(mu_x[i], mu_y[i], color=color_dict[label], label=str(label))

        # TODO: Save the generated figure and include it in your report
        plt.show()

    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha=0.5):
        return alpha * mua + (1 - alpha) * mub

    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings.
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):

        # TODO: Sample 3 pairs of data with different classes
        training_data = self.train_images[0:6, :]

        # TODO: Encode the data in each pair, and take the mean vectors
        mu, sigma = self.encoder(training_data)

        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        interpolation = torch.zeros(33, 2)
        i = 0
        while (i <= 1):
            pair1_inter = self.interpolate_mu(mu[0], mu[1], i)
            pair2_inter = self.interpolate_mu(mu[2], mu[3], i)
            pair3_inter = self.interpolate_mu(mu[4], mu[5], i)
            interpolation[int(i * 10 * 3)] = pair1_inter
            interpolation[int(i * 10 * 3 + 1)] = pair2_inter
            interpolation[int(i * 10 * 3 + 2)] = pair3_inter
            i += 0.1

        # TODO: Along the interpolation, plot the distributions p(x|z_α)
        distribution_arr = np.zeros((33, 28, 28))
        for i in range(interpolation.shape[0]):
            distribution = self.decoder(interpolation[i])
            distribution = array_to_image(distribution.detach().numpy())
            distribution_arr[i] = distribution

        # Concatenate these plots into one figure
        final_figure = concat_images(distribution_arr, 3, 11, padding=3)
        plt.imshow(final_figure)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


args = parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

vae = VAE(args)
vae.train()
vae.visualize_latent_space()
