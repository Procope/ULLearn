import torch
from torch.nn.functional import logsigmoid
from torch.nn import Embedding, Linear, Softplus
from torch.nn.modules.module import Module
from torch.autograd import Variable

class InferenceNet(Module):

    def __init__(self, in_features, out_features):
        super(InferenceNet, self).__init__()

        self.affine1 = Linear(in_features, out_features)
        self.affine2 = Linear(in_features, out_features)

        self.softplus = Softplus()


    def forward(self, input):
        locations = self.affine1(input)
        scales = self.softplus(self.affine2(input))

        return locations, scales



