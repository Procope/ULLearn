import torch
from torch.nn.functional import logsigmoid
from torch.nn import Embedding, Linear
from torch.nn.modules.module import Module
from torch.autograd import Variable

class BayesianSG(Module):

    def __init__(self, vocab_size, latent_size):
        super(BayesianSG, self).__init__()

        self.mus_lookup = Variable(torch.Tensor(vocab_size, latent_size))
        self.sigmas_lookup = Variable(torch.Tensor(vocab_size, latent_size))

        self.affine = nn.Linear(latent_size, vocab_size)
        self.softmax = nn.Softmax()


        self.init_params()


    def init_params(self):
        pass


    def forward(self,
                input  # C x 2*in_features
                ):


        return locations, scales


