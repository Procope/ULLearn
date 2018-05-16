import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn import Embedding, Linear, Softmax, Softplus, ReLU
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence


def multivariate_normal_kl(scale0, scale1, loc0, loc1):
    cov0 = torch.diagflat(scale0 ** 2)
    cov1 = torch.diagflat(scale1 ** 2)

    d0 = MultivariateNormal(loc0, covariance_matrix=cov0)
    d1 = MultivariateNormal(loc1, covariance_matrix=cov1)

    return kl_divergence(d0, d1)


class BayesianSG(Module):

    def __init__(self, vocab_size, embed_dim):
        super(BayesianSG, self).__init__()

        # Sizes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Priors
        self.prior_locs = Embedding(vocab_size, embed_dim)
        self.prior_scales = Embedding(vocab_size, embed_dim)

        # Inference
        self.embeddings = Embedding(vocab_size, embed_dim)
        self.encoder = Linear(2 * embed_dim, 2 * embed_dim)
        self.affine_loc = Linear(2 * embed_dim, embed_dim)
        self.affine_scale = Linear(2 * embed_dim, embed_dim)
        self.std_normal = MultivariateNormal(torch.zeros(embed_dim), torch.eye(embed_dim))

        # Generation
        self.affine_vocab = Linear(embed_dim, vocab_size)

        # Functions
        self.softmax = Softmax(dim=1)
        self.softplus = Softplus()
        self.relu = ReLU()


    def init_embeds():
        """ All embeddings are Xavier-initialised. """
        xavier_uniform_(self.prior_locs.weight)
        xavier_uniform_(self.prior_scales.weight)
        xavier_uniform_(self.embeddings.weight)


    def forward(self,
                center_id,
                context_ids):
        """
        Args: center_id: list of center word ids for positive word pairs.
              context_ids: list of neighbor word ids for positive word pairs.
        """
        center_embed = self.embeddings(Variable(center_id))  # [b,d]
        context_embeds = self.embeddings(Variable(context_ids))  # [b,c,d]

        # Represent words in context
        center_embed = center_embed.unsqueeze(1)  # [b,1,d]
        center_embed = center_embed.repeat(1, context_ids.size()[1], 1)  # [b,c,d]
        encoder_input = torch.cat((center_embed, context_embeds), dim=2)  # [b,c,2d]

        # dimensions_context = list(context_ids.size())
        # center_embed = center_embed.repeat(1, dimensions_context[1], 1)  # [b,c,d]


        # Encode context-aware representations
        h = self.relu(self.encoder(encoder_input))  # [b,c,2d]
        h = torch.sum(h, dim=1)  # [b,2d]

        # Inference step
        loc = self.affine_loc(h)
        scale = self.softplus(self.affine_scale(h))


        # Reparametrization
        epsilon = self.std_normal.sample()
        z = loc + scale * epsilon

        # Prepare arguments of Log-likelihood
        affine_vocab = self.affine_vocab(z)
        categorical = self.softmax(self.affine_vocab(z))  # [b,V]

        # Prepare arguments of KL
        prior_loc = self.prior_locs(Variable(center_id))
        prior_scale = self.softplus(self.prior_scales(Variable(center_id)))



        # Compute ELBO
        kl = []
        for i, _ in enumerate(center_id):
            kl.append(multivariate_normal_kl(scale[i],
                                             prior_scale[i],
                                             loc[i],
                                             prior_loc[i])
            )
        kl = torch.stack(kl)

        lls = []
        for cent_id, cont_ids in enumerate(context_ids):
            ll = 0
            for k in cont_ids:
                ll += torch.log(categorical[
                                torch.tensor(cent_id),
                                torch.tensor(k)
                                ])
            lls.append(ll)

        elbo = torch.tensor(lls) - kl
        loss = torch.mean(- elbo)

        return loss



