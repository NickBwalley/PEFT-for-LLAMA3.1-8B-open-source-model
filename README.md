# Parameter-Efficient Fine-Tuning (PEFT) for Large Language Models. 
## Note: In this project we fine-tuned our quantized open source model LLAMA3.1-8B that outperformed the monstrous ChatGPT 4o with 1.76T+ Params for our product price predictor**

## **Introduction**
Fine-tuning **Large Language Models (LLMs)** from scratch requires extensive compute resources and memory. **Parameter-Efficient Fine-Tuning (PEFT)** provides a lightweight alternative, enabling efficient adaptation of large-scale models while optimizing for memory and compute efficiency.

In this repository, we explore **PEFT techniques**, particularly **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**, applied to **Llama3.1-8B**, with comparisons to **Llama3.1-70B** and **Llama3.1-405B**.

---

## **What is PEFT?**
**PEFT** is a methodology that fine-tunes a small subset of parameters instead of the entire model, reducing computational costs while maintaining high performance. Instead of modifying all the model weights, **PEFT methods** (such as **LoRA** and **QLoRA**) introduce **adapter layers** that learn efficient transformations.

### **Example of PEFT**
Instead of fully fine-tuning a **Llama3.1-70B** model, which is computationally expensive, **LoRA** can modify only a small number of key layers, such as attention weights, to adapt the model efficiently. This reduces GPU memory requirements while achieving performance comparable to full fine-tuning.

---

## **Hyperparameter Tuning**
Hyperparameters are **configuration variables** that control the learning process in deep learning models, such as **learning rate, batch size, weight decay, dropout rates**, etc.

### **Example of Hyperparameter Tuning**
When fine-tuning **Llama3.1-8B**, we may experiment with:
- **Learning rate**: 1e-4 vs 5e-5
- **Batch size**: 8 vs 16
- **LoRA Rank (R)**: 8 vs 16

By optimizing these hyperparameters, we improve model performance while minimizing resource usage.

---

## **Core PEFT Methods**
### **1. LoRA (Low-Rank Adaptation)**
LoRA is a fine-tuning technique that **injects small trainable layers** into **specific parts of the model** (e.g., attention layers) while **freezing the pre-trained weights**. It reduces the number of parameters that require updates, making training **faster and more efficient**.

🔹 **LoRA Example on Llama3.1-8B**
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

lora_config = LoraConfig(
    r=8,  # Low-Rank matrix
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
### **2. QLoRA (Quantized LoRA)**
QLoRA extends LoRA by first quantizing the model weights (e.g., 4-bit or 8-bit precision) before applying LoRA adapters. This drastically reduces VRAM requirements, allowing fine-tuning of Llama3.1-70B on consumer GPUs.

🔹 QLoRA Example on Llama3.1-70B
```
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config
)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

```
### 3. Hyperparameters in PEFT
The key hyperparameters when fine-tuning Llama3.1 models using PEFT:

- ***R (LoRA Rank): Determines the size of the low-rank adapters.***
- ***Alpha (LoRA Scaling Factor): Scales the adapter layers.***
Dropout Rate: Prevents overfitting in fine-tuning.
- ***Target Modules: Specifies which parts of the model to fine-tune (e.g., q_proj, v_proj).***
### Advanced PEFT Techniques
1. Optimizing LLMs: R, Alpha, and Target Modules in QLoRA Fine-Tuning
Choosing R=16 and Alpha=32 can enhance performance while keeping memory usage low.
Targeting only attention layers can maximize fine-tuning efficiency.
2. How to Quantize LLMs (Reducing Model Size with 8-bit Precision)
Quantization reduces model precision, trading accuracy for efficiency. Methods include:

8-bit quantization: Reduces precision but retains most performance.
4-bit quantization: Further reduces memory but may affect accuracy.
🔹 Example: 8-bit Quantization for Llama3.1-8B
```
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", quantization_config=bnb_config)
```
3. Double Quantization & NF4 (Advanced Quantization)
Double Quantization: Further reduces model size by quantizing quantized weights.
NF4 (Normal Float 4-bit Quantization): Optimized for Transformer models, providing better precision than traditional 4-bit quantization.
🔹 Example: Double Quantization with NF4 on Llama3.1-405B
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    double_quant=True,
    nf4=True
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-405B", quantization_config=bnb_config)
```
Performance Comparison (Llama3.1-8B vs 70B vs 405B)

Model	Full Fine-Tuning VRAM	LoRA VRAM	QLoRA VRAM	Double Quantization VRAM
Llama3.1-8B	~100GB	~24GB	~10GB	~8GB
Llama3.1-70B	~800GB	~180GB	~48GB	~36GB
Llama3.1-405B	~5TB	~2TB	~600GB	~450GB
