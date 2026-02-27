# 🚀 Accelerated LLM Inference  
## KV Cache & Speculative Decoding Optimization Study

---

## 🔬 Overview

Investigating algorithmic and systems-level optimizations for autoregressive Large Language Model (LLM) inference are the primary goals of this project.

Implemented and benchmarked decoding strategies as of now:

- Greedy decoding (baseline)
- KV cache optimized decoding
- Naive speculative decoding
- Speculative decoding with KV cache
- Quantization tradeoff analysis

I ahve evaluated how these techniques affect latency, throughput, and scalability across varying prompt lengths.

---

## 🧠 Motivation

LLM inference latency grows with context length because each generated token attends to all previous tokens.

Key challenges:

- Quadratic attention cost
- Memory bandwidth constraints
- GPU kernel launch overhead
- Model size limitations

Modern inference engines use KV caching and speculative decoding to mitigate these bottlenecks.

This project reproduces these techniques from scratch and analyzes their tradeoffs.

---

## ⚙️ Implemented Methods

### 🟢 Greedy Decoding (Baseline)

Sequential token generation without optimization.

**Characteristics:**

- Recomputes attention over the full sequence each step  
- Latency increases with prompt length  
- Serves as baseline for comparison  

---

### 🟡 KV Cache Decoding

Stores past key/value tensors to avoid recomputation.

**Benefits:**

- Generation cost becomes independent of prompt length  
- Dramatically reduces latency for long contexts  
- Used in production inference systems  

---

### 🔴 Naïve Speculative Decoding

Draft model proposes tokens verified by target model, without cache synchronization.

**Observed limitations:**

- Additional coordination overhead  
- Inefficient computation reuse  
- Serves as baseline for speculative approaches  

---

### 🟣 Speculative Decoding with KV Cache

Advanced implementation combining:

- Draft model token proposals  
- Parallel verification by target model  
- Cache synchronization  
- Token acceptance mechanism  

**Expected benefits:**

- Reduced target model invocations  
- Faster generation for long contexts  
- Improved throughput  

---

## 📊 Benchmark Methodology

Experiments measure:

- Latency (seconds)
- Throughput (tokens/sec)
- Speedup vs baseline
- Scaling with prompt length

**Setup:**

- GPU: NVIDIA RTX 3060 (12GB)
- Prompt lengths: short → very long contexts
- Multiple runs averaged to reduce noise
- Fixed output length for fair comparison

---

## 📈 Results

*(To be updated)*

---

## 🔍 Key Findings (Preliminary)

Based on initial experiments:

### KV Cache Scaling

KV caching decouples generation latency from prompt length by reusing past attention states.

---

### Speculative Decoding Behavior

Speculative decoding improves performance when acceptance rates are high and target model calls are reduced.

---

### Quantization Tradeoff Discovery ⭐

4-bit quantization degraded speculative decoding performance due to dynamic dequantization overhead during multi-token verification.

So I have switched back to FP16 weights for faster parallel processing.

---

### Python Overhead Limitation

Draft token generation using Python loops introduced CPU bottlenecks, highlighting the importance of kernel fusion in production systems.

---

## Some Additional Findings

- Algorithmic improvements must align with hardware characteristics  
- Quantization interacts non-trivially with decoding strategies  
- Cache synchronization is critical for correctness  
- Benchmark fairness requires fixed output lengths  

---

## 🛠️ Tech Stack

- PyTorch  
- Hugging Face Transformers  
- BitsAndBytes  
- CUDA GPU acceleration  
- Custom inference engine  

---

