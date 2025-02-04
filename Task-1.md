
#  Bloom
BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total).

Bloom is focused on evaluation of architectural decisions on zero-shot generalization, and does not consider transfer learning. While trainig Bloom, researcher did not consider mixture-of-experts (MoE) and state-space models, due to a lack of widely used GPU-based codebases suitable for training them at scale.

### Key Changest to basic decoder-only transformer architecture
**ALiBi Positional Embeddings:** ALiBi was initially motivated by its ability to extrapolate to longer sequences, but later it was found that it also led to smoother training and better downstream performance even at the original sequence length – outperforming both learned and rotary embeddings.
**Embedding LayerNorm:** Adding LayerNorm after the embedding layer improved training stability in a 104B parameter model, as suggested by bitsandbytes’ StableEmbedding. However, it slightly hurt zero-shot generalization.

A very high fertility on a language compared to a monolingual tokenizer may indicate a degradation on the downstream multilingual performance of the model. The goal was to not degrade fertility rate by 10% when comparing multilingual tokenizer with monolingual tokenizers in corresponding languages. For all experiments, the Hugging Face Tokenizers library was used to design and train the tested tokenizers. Furthermore, Byte Level BPE was used.

# Gemma
Gemma models are a family of lightweight, state-of-the-art open language models developed by Google DeepMind. Designed to be highly efficient and accessible, Gemma models are built on the same research and technology that powers Google's Gemini models, but are optimized for smaller-scale use cases. Gemma 2 models are based on a decoder-only transformer architecture.

Gemma has been trained for large quantities of tokens with distillation in order to simulate training beyond the number of available tokens. Small Language Models are trained with the help of Large Language Model with the help knowledge distillation.

### Model Architecture:
![Architecture Comparison](https://github.com/Darshan-Baslani/IIT-M-assignment/blob/main/images/Gemma.png)
### Key Upgrades
#### For Gemma1:
**Multi-Query Attention(for Gemma):**. Notably, the 7B model uses multi-head attention while the 2B checkpoints use multi-query attention, because multi-query attention works well at small scales.
**RoPE Embeddings:** Rather than using absolute positional embeddings, rotary positional embeddings are used in each layer; embeddings across inputs and outputs are shared to reduce model size. 
**GeGLU Activations:** The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function.
#### For Gemma2:
**Grouped-Query Attention:** GQA with num_groups = 2 has been used, based on ablations showing increased speed at inference time while maintaining downstream performance.
**Post-norm and pre-norm with RMSNorm** Input of each transformer sub-layer, the attention layer and the feedforward layer is normalized, with RMSNorm to stabilize the training.
**Local Sliding Window and Global Attention**. They switch between a local sliding window attention and global attention in every other layer.
**Logit soft-capping** Logits in each attention layer and the final layer are capped such that the value of the logits stays between −soft_cap and +soft_cap.

- Subset of SentencePiece tokenizer of Gemini is used for compatibility. It splits digits, does not remove extra whitespace, and relies on byte-level encodings for unknown tokens. The vocabulary size is 256k tokens.

- Gemma models are not multimodal and are not trained specifically for state-of-the-art multilingual capabilities.

- First, supervised fine-tuning (SFT) is applied on a mix of text-only, English-only synthetic and humangenerated prompt-response pairs. We then apply RLHF on top of these models with the reward model trained on labelled English-only preference data and the policy based on the same prompts as the SFT phase. Finally, we average the models obtained after each phase to improve their overall performance.