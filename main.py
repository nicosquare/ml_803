import torch

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt; 
import tensorflow as tf
import argparse
import random

from torchvision import transforms, datasets
from tqdm import tqdm
from base import BaseVAE
from our_types import *

import PIL
import cv2
from PIL import Image

from src.util import load_baselines_model

plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else  'cpu'
PAC_MAN_SIZE = (84, 84)

print('Using device:', device)

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 7, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=7,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 15, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input[:,0:3,:,:])

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def train(autoencoder, dataloader, epochs=20):

    opt = torch.optim.Adam(autoencoder.parameters())

    best_loss = np.inf

    for epoch in range(epochs):
        
        print(f'Epoch {epoch+1}')

        running_loss = 0.0
        running_kld = 0.0
        running_recons = 0.0

        with tqdm(dataloader, unit='batch') as tepoch:

            for _, (images, labels) in enumerate(tepoch):
                
                images = images.to(device)

                # One hot encoding of the labels

                onehot_label = torch.zeros((labels.shape[0],5), dtype=int)
                onehot_label.scatter_(1, labels.unsqueeze(1), 1)
                repeated_encoding = onehot_label.view(len(labels), 5, 1, 1).repeat(1, 1, 176,176).to(device)

                # Stack encoding with image

                ext_img = torch.cat([images, repeated_encoding], dim=1)

                opt.zero_grad()
                x_hat = autoencoder(ext_img)
                all_losses = autoencoder.loss_function(*x_hat, M_N = 1/len(labels))
                all_losses['loss'].backward()
                opt.step()

                # Save model if loss is better than previous best

                if all_losses['loss'].item() < best_loss:
                    best_loss = all_losses['loss'].item()
                    torch.save({
                        'encoder': autoencoder.encoder.state_dict(),
                        'fc_mu': autoencoder.fc_mu.state_dict(),
                        'fc_var': autoencoder.fc_var.state_dict(),
                        'decoder_input': autoencoder.decoder_input.state_dict(),
                        'final_layer': autoencoder.final_layer.state_dict(),
                        'decoder': autoencoder.decoder.state_dict(),
                        'opt': opt.state_dict(),
                        'loss': all_losses['loss'].item(),
                    },f'models/beta_vae_{autoencoder.latent_dim}.pth')
                
                # Add to the running losses

                running_loss += all_losses['loss'].item()
                running_kld += all_losses['KLD'].item()
                running_recons += all_losses['Reconstruction_Loss'].item()

                # Print training process in the tepoch

                tepoch.set_postfix(loss=all_losses['loss'].item(), kld=all_losses['KLD'].item(), recons=all_losses['Reconstruction_Loss'].item())

        # Print training process in the epoch   

        print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader):.4f} kld: {running_kld/len(dataloader):.4f} recons: {running_recons/len(dataloader):.4f}')

    return autoencoder

def preprocess_frame_ACER(frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            Does NOT apply scaling between 0 and 1 since ACER does not use it
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        if len(frame) == 210:
            frame = frame[0:173, :, :]
        elif len(frame) == 176:
            frame = np.array(Image.fromarray(frame).crop((8, 1, 168, 174)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, PAC_MAN_SIZE, interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

def train_action_encoder(action_encoder, autoencoder, agent, data_loader, epochs=20):

    autoencoder.eval()

    opt = torch.optim.Adam(action_encoder.parameters())

    best_loss = np.inf

    for epoch in range(epochs):

        print(f'Epoch {epoch+1}')

        running_loss = 0.0
        running_labels = 0.0

        with tqdm(data_loader, unit='batch') as tepoch:

            for _, (images, labels) in enumerate(tepoch):

                images = images.to(device)

                # Pick a target action checking that it is not the same as the one chosen by the agent (labels)

                target_actions = torch.tensor([
                    random.choice(
                        [action for action in list(range(0,5)) if action != original_action]
                    ) for original_action in labels
                ]).to(device)

                # One hot encoding of the target actions
                
                onehot_target_action = torch.zeros((target_actions.shape[0],5), dtype=int).to(device)
                onehot_target_action.scatter_(1, target_actions.unsqueeze(1), 1)
                repeated_encoding = onehot_target_action.view(len(target_actions), 5, 1, 1).repeat(1, 1, 176,176).to(device)

                ext_img = torch.cat([images, repeated_encoding], dim=1)

                mu, log_var = action_encoder.encode(ext_img)
                z = action_encoder.reparameterize(mu, log_var)

                agent_actions = []

                # Generate image from the latent representation using trained decoder

                generated_image = autoencoder.decode(z)

                # Crop image to pass it to agent

                for gen_img in generated_image:

                    gen_img = (gen_img.cpu().detach().numpy().reshape(3, 176, 176).transpose(1, 2, 0) * 255).astype(np.uint8)
                    gen_img = preprocess_frame_ACER(gen_img)

                    # We repeat the image because the agent expects a stack of 4 frames

                    stacked_img = np.repeat(gen_img, 4).reshape(1, 84, 84, 4)

                    # Get the action from the agent

                    action = agent.step(stacked_img)[0]

                    agent_actions.append(action.item())

                # Train the action encoder

                opt.zero_grad()
                loss_label = nn.MSELoss()(torch.tensor(np.array(agent_actions, dtype=np.float32), requires_grad=True).to(device).float(), target_actions.float())
                loss_recons = torch.mean(torch.sum(torch.abs(images - generated_image), dim=[1,2,3]))
                loss = loss_label + loss_recons
                loss.backward()
                opt.step()

                # Print training process in the tepoch

                tepoch.set_postfix(loss=loss.item(), loss_label=loss_label.item(), loss_recons=loss_recons.item())

                # Save model if loss is better than previous best

                if loss.item() < best_loss:

                    best_loss = loss.item()

                    torch.save({
                        'encoder': action_encoder.encoder.state_dict(),
                        'fc_mu': action_encoder.fc_mu.state_dict(),
                        'fc_var': action_encoder.fc_var.state_dict(),
                        'opt': opt.state_dict(),
                        'loss': loss.item(),
                    },f'models/beta_vae_action_encoder_{action_encoder.latent_dim}.pth')

                # Add to the running losses

                running_loss += loss.item()
                running_labels += loss_label.item()

        # Print training process in the epoch

        print(f'Epoch {epoch+1} loss: {running_loss/len(data_loader):.4f} loss_label: {running_labels/len(data_loader):.4f}')

    return action_encoder

def validity(target_domain, action_on_counterfactual):
    """
    Calculates the validity of a counterfactual. The counterfactual is valid if the agent chooses the targeted
    action/domain for it.

    :param target_domain: Integer encoded target action/domain.
    :param action_on_counterfactual: The action that the agent chose on the counterfactual frame.
    :return: Bool that indicates the validity.
    """
    return target_domain == action_on_counterfactual

def proximity(original, counterfactual):
    """
    Calculates the proximity of a counterfactual via the L1-norm normalized to range [0, 1].

    :param original: Original numpy frame.
    :param counterfactual: Counterfactual numpy frame.
    :return: The proximity between the counterfactual and the original.
    """
    return 1 - np.linalg.norm((original - counterfactual).flatten(), ord=1) / (original.size * 255)

def sparsity(original, counterfactual):
    """
    Calculates the sparsity of a counterfactual via the L0-norm normalized to range [0, 1].

    :param original: Original numpy frame.
    :param counterfactual: Counterfactual numpy frame.
    :return: The sparsity between the counterfactual and the original.
    """
    return 1 - np.linalg.norm((original - counterfactual).flatten(), ord=0) / original.size

def evaluate_action_encoder(action_encoder, autoencoder, agent, data_loader):

    action_encoder.eval()
    autoencoder.eval()

    validities = []
    sparsities = []
    proximities = []

    with tqdm(data_loader, unit='batch') as tepoch:

        for _, (images, labels) in enumerate(tepoch):

            images = images.to(device)

            # Pick a target action checking that it is not the same as the one chosen by the agent (labels)

            target_actions = torch.tensor([
                random.choice(
                    [action for action in list(range(0,5)) if action != original_action]
                ) for original_action in labels
            ]).to(device)

            # One hot encoding of the target actions
            
            onehot_target_action = torch.zeros((target_actions.shape[0],5), dtype=int).to(device)
            onehot_target_action.scatter_(1, target_actions.unsqueeze(1), 1)
            repeated_encoding = onehot_target_action.view(len(target_actions), 5, 1, 1).repeat(1, 1, 176,176).to(device)

            ext_img = torch.cat([images, repeated_encoding], dim=1)

            mu, log_var = action_encoder.encode(ext_img)
            z = action_encoder.reparameterize(mu, log_var)

            agent_actions = []

            # Generate image from the latent representation using trained decoder

            generated_image = autoencoder.decode(z)

            # Crop image to pass it to agent

            for ix_img, gen_img in enumerate(generated_image):

                proximities.append(proximity(images[ix_img].cpu().detach().numpy(), gen_img.cpu().detach().numpy()))
                sparsities.append(sparsity(images[ix_img].cpu().detach().numpy(), gen_img.cpu().detach().numpy()))

                gen_img = (gen_img.cpu().detach().numpy().reshape(3, 176, 176).transpose(1, 2, 0) * 255).astype(np.uint8)
                gen_img = preprocess_frame_ACER(gen_img)

                # We repeat the image because the agent expects a stack of 4 frames

                stacked_img = np.repeat(gen_img, 4).reshape(1, 84, 84, 4)

                # Get the action from the agent

                action = agent.step(stacked_img)[0]

                validities.append(validity(target_actions[ix_img], action.item()).item())

                tepoch.set_postfix(validity=np.mean(validities), sparsity=np.mean(sparsities), proximity=np.mean(proximities))

        # Print final metrics

        log = f'Validities: {np.mean(validities)}\nSparsities: {np.mean(sparsities)}\nProximities: {np.mean(proximities)}'

        print(log)

def main():

    # Process arguments

    parser = argparse.ArgumentParser(description='Train a Beta-VAE model')

    parser.add_argument('--latent_dim', type=int, default=2, help='latent dimension of the model')
    parser.add_argument('--beta', type=int, default=4, help='beta parameter of the model')
    parser.add_argument('--gamma', type=float, default=1000., help='gamma parameter of the model')
    parser.add_argument('--max_capacity', type=int, default=25, help='max capacity parameter of the model')
    parser.add_argument('--Capacity_max_iter', type=int, default=1e5, help='capacity max iter parameter of the model')
    parser.add_argument('--loss_type', type=str, default='H', help='loss type parameter of the model')
    parser.add_argument('--trained_vae', action='store_true', help='no need to train the model')
    parser.add_argument('--epochs_vae', type=int, default=5, help='number of epochs to train the model')
    parser.add_argument('--trained_action_encoder', action='store_true', help='no need to train the model')
    parser.add_argument('--epochs_action_encoder', type=int, default=5, help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of the model')

    args = parser.parse_args()

    # Set seeds

    init_seeds()

    # Define the path to your image folder
    data_path = "res/datasets/PacMan_FearGhost_cropped_5actions_Unique/train"

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),  # You can adjust the size as needed
        transforms.ToTensor(),
    ])

    # Create the ImageFolder dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Create the data loader
    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train Beta-VAE

    latent_dims = args.latent_dim
    nb_classes = 5

    autoencoder = BetaVAE(in_channels=(3 + nb_classes), latent_dim=latent_dims, loss_type='H', beta=2).to(device) # GPU

    data = data_loader

    if not args.trained_vae:
        autoencoder = train(autoencoder, data, epochs=args.epochs_vae)

    else:

        # Load the model from models folder

        checkpoint = torch.load(f'models/beta_vae_{latent_dims}.pth')

        autoencoder.encoder.load_state_dict(checkpoint['encoder'])
        autoencoder.fc_mu.load_state_dict(checkpoint['fc_mu'])
        autoencoder.fc_var.load_state_dict(checkpoint['fc_var'])
        autoencoder.decoder_input.load_state_dict(checkpoint['decoder_input'])
        autoencoder.final_layer.load_state_dict(checkpoint['final_layer'])
        autoencoder.decoder.load_state_dict(checkpoint['decoder'])

    # Load the model trained with the baselines library

    agent_file = "res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3"

    agent = load_baselines_model(agent_file, num_actions=5, num_env=1)

    # Train action encoder

    action_encoder = BetaVAE(in_channels=(3 + nb_classes), latent_dim=latent_dims, loss_type='H', beta=2).to(device) # GPU

    if not args.trained_action_encoder:

        action_encoder = train_action_encoder(action_encoder, autoencoder, agent, data_loader, epochs=args.epochs_action_encoder)

        # Preload the training from the autoencoder for the encoder

        action_encoder.encoder.load_state_dict(autoencoder.encoder.state_dict().copy())
        action_encoder.fc_mu.load_state_dict(autoencoder.fc_mu.state_dict().copy())
        action_encoder.fc_var.load_state_dict(autoencoder.fc_var.state_dict().copy())

    else:

        # Load the model from models folder

        checkpoint = torch.load(f'models/beta_vae_action_encoder_{latent_dims}.pth')

        action_encoder.encoder.load_state_dict(checkpoint['encoder'])
        action_encoder.fc_mu.load_state_dict(checkpoint['fc_mu'])
        action_encoder.fc_var.load_state_dict(checkpoint['fc_var'])

    # Evaluate action encoder

    evaluate_action_encoder(action_encoder, autoencoder, agent, data_loader)


if __name__ == "__main__":
    main()
