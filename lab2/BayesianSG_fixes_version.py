import torch
from torch.nn import Embedding, Linear, Softmax, Softplus, ReLU
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
import torch.distributions as distributions
import numpy as np


class BayesianSG(Module):

    def __init__(self, vocab_size, embed_dim, unigram_probs):
        super(BayesianSG, self).__init__()

        # Sizes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Priors
        self.prior_means = Embedding(vocab_size, embed_dim)
        self.prior_vars = Embedding(vocab_size, embed_dim)
        self.prior_means.weight.requires_grad = False
        self.prior_vars.weight.requires_grad = False

        # Inference
        self.embeddings = Embedding(vocab_size, embed_dim)
        #self.embeddings.weight.requires_grad = False

        self.encoder = Linear(2 * embed_dim, 2 * embed_dim)
        self.affine_mean = Linear(2 * embed_dim, embed_dim)
        self.affine_var = Linear(2 * embed_dim, embed_dim)
        self.std_normal = MultivariateNormal(torch.zeros(embed_dim), torch.eye(embed_dim))

        # Generation
        self.affine_vocab = Linear(embed_dim, vocab_size)
        self.unigram_probs = unigram_probs

        # Functions
        self.softmax = Softmax(dim=1)
        self.softplus = Softplus()
        self.relu = ReLU()

    def forward(self,
                center_id,
                context_ids):
        """
        Args: center_id: list of center word ids for positive word pairs.
              pos_context_id: list of neighbor word ids for positive word pairs.
        """
        center_embed = self.embeddings(Variable(center_id))  # [b,d] #todo: move Variable to preproc
        context_embeds = self.embeddings(Variable(context_ids))  # [b,c,d]
        print("context size", context_embeds.size())

        # Represent words in context
        center_embed = center_embed.unsqueeze(1)  # [b,1,d]

        dimensions_context = list(context_ids.size())
        center_embed = center_embed.repeat(1, dimensions_context[1], 1)  # [b,c,d]

        encoder_input = torch.cat((center_embed, context_embeds), dim=2)  # [b,c,2d]
        print("encoder input", encoder_input.size())

        # Encode context-aware representations
        h = self.relu(self.encoder(encoder_input))  # [b,c,2d]
        h = torch.sum(h, dim=1)
        print("h", h.size())

        # Inference step
        mean = self.affine_mean(h)
        var = self.softplus(self.affine_var(h))
        print("mean", mean.size(), "var", var.size())
        # Reparametrization
        epsilon = self.std_normal.sample()
        z = mean + torch.exp(var / 2.) * epsilon

        print("z", z.size())

        # Prepare arguments of the loss
        # context_unigram_p = self.unigram_probs[context_ids]  # [b,c]
        affine_vocab = self.affine_vocab(z)
        print("affine vocab", affine_vocab.size())
        categorical = self.softmax(self.affine_vocab(z))  # [b,c,V]
        print("cat", categorical.size())
        # todo: multiply by center_embed?
        prior_mean = self.prior_means(Variable(center_id))
        prior_var = self.softplus(self.prior_vars(Variable(center_id)))

        # Compute the loss
        # kl = -0.5 + torch.log(prior_var / var) + (0.5 * (var ** 2 + (mean - prior_mean) ** 2) / (prior_var ** 2))
        print("var", var.size())
        a = torch.diag(var[1] ** 2)
        print("cov", a.size())

        def kl_div(s0, s1, m_0, m_1):
            # u,l are cov matrices
            # m_1 and m_2 are mean vectors
            # print(s0.detach().numpy())

            s0 = np.diag(s0.detach().numpy())

            s1 = np.diag(s1.detach().numpy())

            m_0 = m_0.detach().numpy()

            m_1 = m_1.detach().numpy()

            kl = 0.5 * (np.trace(np.matmul(np.linalg.inv(s1), s0))
                        + np.matmul(np.matmul(np.transpose(m_1 - m_0), np.linalg.inv(s1)), (m_1 - m_0)) - s0.shape[0]
                        + np.log(np.linalg.det(s1) / np.linalg.det(s0)))
            return kl

        kl = []

        for i in range(self.embed_dim):

            kl.append(kl_div(var[i], prior_var[i], mean[i], prior_mean[i]))

            #kl = -0.5 + torch.log(prior_var / var) + (0.5 * (var ** 2 + (mean - prior_mean) ** 2) / (prior_var ** 2))
            #posterior = MultivariateNormal(mean[i], torch.diag(var[i])**2 )
            #prior = MultivariateNormal(prior_mean[i], torch.diag(prior_var[i])**2 )
            #print(torch.diag(var[i]) ** 2)
            #print(torch.diag(prior_var[i]) ** 2)
            #kl.append(torch.distributions.kl.kl_divergence(posterior, prior) * len(mean))

        kl = torch.tensor(kl)

        print("kl", kl.size())

        reconstruction_errors = []  # torch.zeros(self.embed_dim, self.embed_dim)

        for cent_id, cont_ids in zip(torch.tensor(range(len(center_id))), context_ids):

            reconstruction_error = 0
            for k in cont_ids:
                a = torch.log(categorical[Variable(cent_id), Variable(k)])
                print(a)
                reconstruction_error += torch.log(categorical[Variable(cent_id), Variable(k)])
            reconstruction_errors.append(reconstruction_error)

        loss = torch.tensor(reconstruction_errors) - kl

        return torch.sum(loss)
