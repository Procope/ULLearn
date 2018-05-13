import torch
from torch.nn import Embedding, Linear, Softmax, Softplus, ReLU
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.distributions import MultivariateNormal



class BayesianSG(Module):

    def __init__(self, vocab_size, embed_dim, unigram_probs):
        super(BayesianSG, self).__init__()

        # Sizes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Priors
        self.prior_means = Embedding(vocab_size, embed_dim)
        self.prior_vars  = Embedding(vocab_size, embed_dim)

        # Inference
        self.embeddings = Embedding(vocab_size, embed_dim)
        self.encoder = Linear(2*embed_dim, 2*embed_dim)
        self.affine_mean = Linear(2*embed_dim, embed_dim)
        self.affine_var = Linear(2*embed_dim, embed_dim)
        self.std_normal = distributions.MultivariateNormal(torch.zeros(embed_dim), torch.eye(embed_dim))

        # Generation
        self.affine_vocab = Linear(embed_dim, vocab_size)
        self.unigram_probs = unigram_probs

        # Functions
        self.softmax = Softmax(dim=2)
        self.softplus = Softplus()
        self.relu = ReLU()


    def forward(self,
                center_id,
                context_ids):
        """
        Args: center_id: list of center word ids for positive word pairs.
              pos_context_id: list of neighbor word ids for positive word pairs.
        """
        center_embed = embeddings(Variable(center_id))  # [b,d] #todo: move Variable to preproc
        context_embeds = embeddings(Variable(context_ids))  # [b,c,d]

        # Represent words in context
        center_embed = center_embed.unsqueeze(1)  # [b,1,d]
        center_embed = center_embed.repeat(1, len(context_ids), 1)  # [b,c,d]
        encoder_input = torch.cat((center_embed, context_embeds), dim=2)  # [b,c,2d]

        # Encode context-aware representations
        h = self.relu(self.encoder(encoder_input)) # [b,c,2d]
        h = torch.sum(h, dim=1)

        # Inference step
        mean = self.affine_mean(h)
        var = self.softplus(self.affine_var(h))

        # Reparametrization
        epsilon = self.std_normal.sample()
        z = mean + torch.exp(var / 2.) * epsilon

        # Prepare arguments of the loss
        # context_unigram_p = self.unigram_probs[context_ids]  # [b,c]
        categorical = self.softmax(self.affine_vocab(z))  # [b,c,V]

        # todo: multiply by center_embed?
        prior_mean = self.prior_means[center_id]
        prior_var = softplus(self.prior_sigma[center_id])

        # Compute the loss
        kl = -0.5 + torch.log(prior_var / var) + (0.5 * (var ** 2 + (mean - prior_mean) ** 2) / (prior_var ** 2))

        reconstruction_error = 0
        for idx in context_ids:
            reconstruction_error += torch.log(categorical[i])

        loss = reconstruction_error - kl

        return loss




