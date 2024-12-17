# UCloud Workshop 11-12-2024

**Advanced AI Tool Development on DeiC Interactive HPC UCloud**

Workshop delivered on **December 11th, 2024**

## Table of Contents

- [Overview](#overview)
- [Contents](#contents)
- [Getting Started](#getting-started)

## Overview

This collection of Jupyter notebooks is designed to guide participants through the comprehensive process of fine-tuning, deploying, and performance testing Large Language Models (LLMs) on UCloud's Interactive High-Performance Computing (HPC) infrastructure. Leveraging powerful tools like the NeMo Framework, Triton Inference Server, and TensorRT-LLM, these notebooks provide hands-on experience in optimizing and deploying state-of-the-art AI models.

**Key Learning Objectives:**

- Fine-tune Llama models using the NeMo Framework.
- Optimize models for deployment with TensorRT-LLM.
- Deploy models on Triton Inference Server for scalable and efficient inference.
- Conduct performance profiling to evaluate deployment efficacy.

## Contents

This repository comprises three primary Jupyter notebooks, each focusing on different stages of the AI model development and deployment pipeline:

1. **[Fine-Tuning Llama-3.1-8B with NeMo Framework](notebooks/llama3.1-8B-lora-nemo.ipynb)**
   - **Description:** Learn how to fine-tune the Llama-3.1-8B model using the NeMo Framework, incorporating Low-Rank Adaptation (LoRA) techniques for efficient parameter tuning.

2. **[Fine-Tuning Llama-3.1-70B with NeMo Framework](notebooks/llama3.1-70B-lora-nemo.ipynb)**
   - **Description:** Extend your fine-tuning skills to the larger Llama-3.1-70B model, exploring scalability and optimization strategies within the NeMo environment.

3. **[Deploying Llama-3.3-70B with Triton Inference Server](notebooks/llama3.3-70B-triton.ipynb)**
   - **Description:** Dive into the deployment process of the optimized Llama-3.3-70B model using Triton Inference Server, enhancing inference performance with TensorRT and conducting comprehensive performance profiling.


## Getting Started

Follow the steps below to set up your environment and begin using the notebooks.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **UCloud Account:**  
  Sign up for a [UCloud account](https://cloud.sdu.dk) to access Interactive HPC services.

- **Hugging Face Token:**  
  Obtain a [Hugging Face token](https://huggingface.co/settings/tokens) to access and download model repositories.

- **NeMo Framework:**  
  Launch an instance of the [NeMo Framework](https://docs.cloud.sdu.dk/Apps/nemo.html) version `24.07` with **4 NVIDIA H100 GPUs** (machine type `u3-gpu-4`).

- **Triton Inference Server:**  
  Start an instance of the [Triton Inference Server (TRT-LLM)](https://docs.cloud.sdu.dk/Apps/triton.html) version `24.08` with **4 NVIDIA H100 GPUs** (machine type `u3-gpu-4`).
