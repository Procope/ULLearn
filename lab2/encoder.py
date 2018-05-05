import torch
from torch.nn.functional import logsigmoid
from torch.nn import Embedding, Linear
from torch.nn.modules.module import Module
from torch.autograd import Variable

class Encoder(Module):

    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()

        self.M = nn.Linear(2*in_features, out_features, bias=False)
        self.relu = nn.ReLU()

        self.U = nn.Linear(out_features, out_features)
        self.W = nn.Linear(out_features, out_features)  # todo: nn.Linear(out_features, 1) ?  Equations 5,6
        self.softplus = nn.Softplus()

        self.init_params()


    def init_params(self):
        pass
        # initrange = 0.5 / self.embed_dim
        # self.input_embeds.weight.data.uniform_(-initrange, initrange)
        # self.output_embeds.weight.data.uniform_(-0, 0)


    def forward(self,
                input  # C x 2*in_features
                ):

        x = self.M(input)
        x = self.relu(x)
        h = torch.sum(x, dim=0)

        locations = self.U(x)
        scales = self.softplus(self.W(x))  # todo: exp this?

        return locations, scales


