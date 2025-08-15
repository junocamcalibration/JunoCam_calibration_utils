
import torch
import torch.nn as nn
from torchvision import models
import scipy
import numpy as np


class FIDCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        self.inception_model = self._get_inception_model()

    def _get_inception_model(self):
        inception_model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1').to(self.device)
        inception_model.avgpool.register_forward_hook(self.output_hook)
        inception_model.eval()
        return inception_model


    def output_hook(self, module, input, output):
        # N x 2048 x 1 x 1
        self.avgpool = output


    def _get_activations(self, images):
        with torch.no_grad():
            #activations = self.inception_model(images)
            self.inception_model(images)
            activations = self.avgpool # N x 2048 x 1 x 1
        return activations[:, :, 0, 0].cpu().numpy()


    def calculate_fid(self, real_images, generated_images, batch_size=1):
        real_activations = []
        generated_activations = []

        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i + batch_size].to(self.device)
            real_activations.extend(self._get_activations(batch))

        for i in range(0, len(generated_images), batch_size):
            batch = generated_images[i:i + batch_size].to(self.device)
            generated_activations.extend(self._get_activations(batch))

        real_activations = np.asarray(real_activations)
        generated_activations = np.asarray(generated_activations)

        real_mu, real_sigma = self._calculate_statistics(real_activations)
        generated_mu, generated_sigma = self._calculate_statistics(generated_activations)

        return self._calculate_frechet_distance(real_mu, real_sigma, generated_mu, generated_sigma)


    def _calculate_statistics(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma


    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        '''
        The Frechet distance between two multivariate Gaussians 
        X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        '''
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
