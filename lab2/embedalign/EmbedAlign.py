import torch
from torch.autograd import Variable
from torch.nn.functional import logsigmoid
from torch.nn.modules.module import Module
from torch.distributions import MultivariateNormal
from torch.nn import Embedding, Linear, CrossEntropyLoss, Softmax

from inference import InferenceNet


class EmbedAlign(Module):

    def __init__(self, vocab_size_l1, vocab_size_l2, embed_dim, context):
        super(EmbedAlign, self).__init__()

        # Sizes
        self.vocab_size_l1 = vocab_size_l1
        self.vocab_size_l2 = vocab_size_l2
        self.embed_dim = embed_dim

        self.context = context  # boolean: whether to encode words in context
        if self.context:
            self.h_dim = 2 * embed_dim  # 'd' in comments
        else:
            self.h_dim = embed_dim

        # Inference
        self.embeddings = Embedding(vocab_size_l1, embed_dim)
        self.inference_net = InferenceNet(self.h_dim, embed_dim)

        # Generation
        self.affine_l1 = Linear(embed_dim, vocab_size_l1)
        self.affine_l2 = Linear(embed_dim, vocab_size_l2)

        # Functions
        self.softmax = Softmax(dim=2)
        self.cross_entropy = CrossEntropyLoss()


    def forward(self, batch_l1, batch_l2):

        batch_size = batch_l1.size()[0]  # 'b' in comments
        m = batch_l1.size()[1]  # longest English sentence in batch
        n = batch_l2.size()[1]  # longest French sentence in batch

        embedded_l1 = self.embeddings(batch_l1)  # [b,m,d]

        l1_mask = torch.sign(batch_l1).float()
        l2_mask = torch.sign(batch_l2).float()

        # Exact sentence lengths
        l1_sent_lengths = torch.sum(l1_mask, dim=1)
        l1_sent_lengths = torch.unsqueeze(l1_sent_lengths, dim=1)

        # Alignment uniform probabilities
        align_probs = l1_mask / l1_sent_lengths.float()
        align_probs = torch.unsqueeze(align_probs, dim=1)
        align_probs = align_probs.repeat(1, n, 1)

        # Encoding words in context.
        # This is an alternative to the BiLSTM of the original paper.
        if self.context:
            sums = torch.sum(embedded_l1, dim=1)
            sums = sums.unsqueeze(1).repeat(1, m, 1)
            context = sums - embedded_l1
            context /= m - 1
            h = torch.cat((embedded_l1, context), dim=2)  # [b,m,2d]
        else:
            h = embedded_l1  # [b,m,d]


        # Inference
        z_loc, z_scale = self.inference_net(h)  # [b,m,d], [b,m,d]

        # Reparametrisation trick
        std_normal = MultivariateNormal(torch.zeros(self.embed_dim), torch.eye(self.embed_dim))
        epsilon = std_normal.sample()
        z = z_loc + z_scale * epsilon

        # Obtain categorical distributions
        logits_l1 = self.affine_l1(z)  # [b,m,V_l1]
        cat_l1 = self.softmax(logits_l1)

        logits_l2 = self.affine_l2(z)  # [b,m,V_l2]
        cat_l2 = self.softmax(logits_l2)

        # Marginalise out alignment probabilities
        p_l2_zx = torch.bmm(align_probs, cat_l2)  # [b,n,V_l2]


        # Compute loss

        # L1 and L2 negative log-likelihoods
        cross_entropy_l1 = self.cross_entropy(logits_l1.permute([0, 2, 1]), batch_l1)  # [b,m]
        cross_entropy_l1 = torch.sum(cross_entropy_l1 * l1_mask, dim=1)  # [b]
        cross_entropy_l1 = torch.mean(cross_entropy_l1, dim=0)  # []

        cross_entropy_l2 = self.cross_entropy(p_l2_zx.permute([0, 2, 1]), batch_l2)
        cross_entropy_l2 = torch.sum(cross_entropy_l2 * l2_mask, dim=1)  # [b]
        cross_entropy_l2 = torch.mean(cross_entropy_l2, dim=0)  # []

        # KL divergence
        z_var = z_scale ** 2
        kl_z = -0.5 * (1 + torch.log(z_var) - z_loc ** 2 - z_var)  # [b,m,d]
        kl_z = torch.sum(kl_z, dim=2)  # [b, m]
        kl_z = torch.sum(kl_z * l1_mask, dim=1)  # [b]
        kl_z = torch.mean(kl_z, dim=0)  # []

        # Loss = - ELBO
        loss = cross_entropy_l1 + cross_entropy_l2 + kl_z


        # Compute Alignment Error Rate

        # Alignment Error Rate for L1
        predicted_batch_l1 = torch.argmax(cat_l1)
        acc_l1 = (predicted_batch_l1 == batch_l1).float()
        acc_l1 = acc_l1 * l1_mask

        acc_l1_correct = torch.sum(acc_l1)
        acc_l1_total = torch.sum(l1_mask)
        acc_l1 = acc_l1_correct / acc_l1_total

        # Alignment Error Rate for L2
        predicted_batch_l2 = torch.argmax(p_l2_zx)
        acc_l2 = (predicted_batch_l2 == batch_l2).float()
        acc_l2 = acc_l2 * l2_mask

        acc_l2_correct = torch.sum(acc_l2)
        acc_l2_total = torch.sum(l2_mask)
        acc_l2 = acc_l2_correct / acc_l2_total


        return loss, acc_l1, acc_l2
