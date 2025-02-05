Link to the notebook: [here](https://colab.research.google.com/drive/1CzHfjv3hmnoaboI39Hs4uNCCYk2YcNuT?usp=sharing)

# Introduction

This notebook explores techniques for optimizing the loading and inference of large language models (LLMs), specifically the `google/gemma-7b` model, within the Google Colab environment. It demonstrates the use of **quantization** and **model sharding** to reduce memory footprint and improve inference speed. The notebook showcases the impact of these optimizations on both English and Hindi text generation tasks, highlighting the feasibility of deploying large LLMs on resource-constrained platforms like Colab. 8bit Post Training Quantization is used.

# Techniques used
**Quantization:** This process involves converting high-precision numerical representations (like 32-bit floating-point numbers) into lower-precision formats (such as 8-bit floating-point numbers), which can significantly speed up inference times and decrease power consumption. Post Training Quantization is used with of conversion of 32-bit floating points to 8-bit floating points.

**Model Sharding:** Model sharding is a distributed training technique that involves breaking down a large model into smaller, manageable segments or "shards." This allows the model to be processed across multiple devices or nodes simultaneously, enhancing resource utilization and scalability.

# Raw Data

| Model Parameters | Peak RAM Usage | Peak VRAM Usage | Inference Time(En/Hi) | Model Load Time |
|----------|----------| ----------| ----------| ----------| 
| Quant=True, Sharding=True, batch=1 | 4.9GB / 12.7GB | 11.1GB / 15.0GB | (23/17)s | 112s |
| Quant=True, Sharding=True, batch=16 | 5.0GB / 12.7GB | 14.1GB / 15.0GB | (32/33)s | 112s |
| Quant=False, Sharding=True, batch=1 | 11.1GB / 12.7GB | 11.9GB / 15.0GB | (2880/-)s | 496s |
| Quant=False, Sharding=True, batch=16 | 11.1GB / 12.7GB | 13.9GB / 15.0GB | (3079/-)s | 496s |
| Quant=True, Sharding=False, batch=1 | 11.1 / 12.7GB | - / 15.0GB | (20/-)s | 100s |
| Quant=True, Sharding=False, batch=16 | 11.1 / 12.7GB | - / 15.0GB | (33/-)s | 100s |

- The quantized models (Quant=True) are running inference in the low tens of seconds (23/17s for batch=1 and 32/33s for batch=16) compared to the non-quantized ones taking a ludicrous 2880–3079 seconds.
- Loading time follows the same pattern—112s for quantized vs. 496s for non-quantized.
- Compare “Quant=True, Sharding=True, batch=1” (112s load, 23s inference) with “Quant=True, Sharding=False, batch=1” (100s load, 20s inference). Sharding is adding a bit of overhead—about 12s on load and 3s on inference. This could be because sharding doesn't store all of its memory in VRAM, to load up the required memory from ROM to VRAM takes time.
- But Sharding saves us RAM, compare "Quant=True, Sharding=False, batch=1"(11.1GB RAM usage) with "Quant=True, Sharding=True, batch=1"(4.9GB RAM usage)
- With batch=16, the differences narrow a bit in inference time (33s regardless of sharding for quantized models). Sharding doesn’t drastically affect larger batch processing in this case.
- Quantization (8-bit) reduces peak RAM usage by 55-56% (11.1GB → 4.9-5.0GB). This aligns with expectations since 8-bit weights require 4x less memory than 32-bit floats.
- Larger batches increase inference times unexpectedly:
	-   Quant=True: +39% (23s→32s)
	-   Quant=False: +6.9% (2880s→3079s)
# Q&A
**Q1** How do these compressed models generate text? Can they generate text as well as their non-compressed versions?
-> The Compressed models generated text faster than non-compressed version. This is because 8-bit weights are faster to compute than 32-bit weights. In my experience, the non-compressed version generates better text, in my Hindi prompt-"एक मछली, पानी में गई," the compressed version started generating English tokens, but uncompressed generated only Hindi tokens.

**Q2** Does generation speed improve or degrade when compressing?
-> As mentioned above, the generation speed increases when compressing. This is because 8-bit weights are faster to compute than 32-bit weights. 
**Q3** Can we increase batch sizes when compressing? How does batching affect generation speed?
-> Yes we can increase batch sizes as demonstrated in the notebook. Because of Batching the inference time grows unexpectedly. Also, the VRAM needs more compute while dealing with batches. This is also quite visible in the above table. Although batching doesn't affect RAM in our example.
**Q4** What are the largest models you can fit on the colab GPUs? What tricks did you use?
-> I have used Quantization and Model Sharding. I tried loading 27B parameters model but it couldn't load it as the model size was 27B * 4 = ~108GB and google colab doesn't give that much storage. Although even if colab didn't have limited ROM, it has limited RAM and VRAM. Even if the model was loaded it wouldn't have been able to load into the RAM(which is mere 12GB).