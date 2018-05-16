## Learning Word Representations

In this lab, we implement and train three word representation models: one, **Skip-gram**, trained for maximum likelihood, and two latent variable models trained by variational inference: **Bayesian Skip-gram** and **EmbedAlign**.  Skip-gram is trained discriminatively by having a central word predict context words in a window surrounding it. Bayesian Skip-gram introduces stochastic latent embeddings but does not change the discriminative nature of the training procedure. EmbedAlign introduces stochastic latent embeddings as well as a latent alignment variable and learns by generating translation data. We compare the performance of these three models on the lexical substitution task.

---------------

Setting up:
- `cd ULLearn/lab2`
- `python setup.py install`

Training:
- `python skipgram/train_skipgram.py --batch_size 100 --n_batches 50 --lr 0.001`
- `python embedalign/train_embedalign.py --batch_size 100 --n_batches 50 --context`
- `python bsg/train_bsg.py --batch_size 100 --n_batches 50 --epochs 50`

Evaluating:
- `python utils/lexical_substitution.py --model skipgram --w2i <w2i.p> --model_path <wmbeds.txt> --skipgram_mode add`
- `python utils/lexical_substitution.py --model embedalign --w2i <w2i.p> --model_path <model.p>`
- `python utils/lexical_substitution.py --model bsg --w2i <w2i.p> --model_path <model.p>`
- `python embedalign/test_embedalign.py --path <model.p>`

