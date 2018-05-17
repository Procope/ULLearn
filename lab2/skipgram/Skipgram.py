import torch
from torch.nn.functional import logsigmoid
from torch.nn import Embedding
from torch.nn.modules.module import Module
from torch.autograd import Variable


class Skipgram(Module):

    def __init__(self, vocab_size, embed_dim):
        super(Skipgram, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.input_embeds = Embedding(vocab_size, embed_dim, sparse=False)
        self.output_embeds = Embedding(vocab_size, embed_dim, sparse=False)
        self.init_emb()


    def init_emb(self):
        initrange = 0.5 / self.embed_dim
        self.input_embeds.weight.data.uniform_(-initrange, initrange)
        self.output_embeds.weight.data.uniform_(-0, 0)


    def forward(self, center_id, pos_context_id, neg_context_ids):
        """
        Args: center_id: list of center word ids for positive word pairs.
              pos_context_id: list of neighbor word ids for positive word pairs.
              neg_context_ids: list of neighbor word ids for negative word pairs.
        """

        # Obtain embeddings for all word ids
        center_embed = self.input_embeds(Variable(torch.LongTensor(center_id)))
        pos_context_embed = self.output_embeds(Variable(torch.LongTensor(pos_context_id)))
        neg_context_embeds = self.output_embeds(Variable(torch.LongTensor(neg_context_ids)))


        # Compute loss
        pos_score = (center_embed * pos_context_embed).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_loss = logsigmoid(pos_score)

        neg_score = torch.bmm(neg_context_embeds, center_embed.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_loss = logsigmoid(-1 * neg_score)

        loss = torch.sum(pos_loss) + torch.sum(neg_loss)

        return -loss
