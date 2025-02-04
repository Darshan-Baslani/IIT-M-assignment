# Bloom

BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of
sources in 46 natural and 13 programming languages (59 in total). BLOOM is a 176 billion parameter language model. Bloom uses language modeling. Language modeling refers to the task of modeling the probability of a sequence of tokens in a
text, where a token is a unit of text. In this work we model the joint probability of tokens in a text as:
\[
p(x) = p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}) \quad (1)
where x is a sequence of tokens, xt is the t_th token, and x < t is the sequence of tokens preceding xt. This approach is referred to as autoregressive language modeling and can be seen as iteratively predicting the probability of the next token.

Bloom is focused on evaluation of architectural decisions on zero-shot generalization, and does not consider transfer learning. While trainig Bloom, researcher did not consider mixture-of-experts (MoE), due to a lack of widely used GPU-based codebases suitable for training them at scale. Similarly, they also did not consider state-space models.
Beyond choosing an architecture and pretraining objective, a number of changes to the original Transformer architecture have been proposed. For example, alternative positional embedding schemes or novel activation functions