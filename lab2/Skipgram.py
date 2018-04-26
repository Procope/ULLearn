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


    def forward(self,
                center_id,
                pos_context_id,
                neg_context_ids):
        """
        Forward process. As pytorch designed, all variables must be batch format,
        so all input of this method is a list of word id.

        Args: center_id: list of center word ids for positive word pairs.
              pos_context_id: list of neighbor word ids for positive word pairs.
              neg_context_ids: list of neighbor word ids for negative word pairs.
        """

        losses = []

        center_embed = self.input_embeds(Variable(torch.LongTensor(center_id)))
        pos_context_embed = self.output_embeds(Variable(torch.LongTensor(pos_context_id)))


        score = (center_embed * pos_context_embed).squeeze()  # elementwise multiplication: batch_size x embed_dim
        score = torch.sum(score, dim=1)

        score = logsigmoid(score)
        losses.append(sum(score))

        neg_context_embeds = self.output_embeds(Variable(torch.LongTensor(neg_context_ids)))
        neg_score = torch.bmm(neg_context_embeds, center_embed.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)

        neg_score = logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))

        return -1 * sum(losses)
