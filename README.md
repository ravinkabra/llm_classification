# 🧠 Bank Consumer Complaints AI: Fine-Tuning, Classification & Agentic Triage

This project presents a comprehensive pipeline for understanding and responding to consumer banking complaints, combining:

1. **Transformer Model Comparison**: Fine-tuning and evaluating BERT and LLaMA variants on classification tasks
2. **Agentic AI Triage System**: Retrieval-Augmented Generation using LangChain, FAISS, and Gemini Flash to auto-triage incoming complaints

---

## 🎯 Project Objectives

- Evaluate how fine-tuning smaller models compares to scaling larger, general-purpose LLMs
- Build an agentic system that retrieves similar cases, reasons over complaint context, and generates structured triage outputs
- Demonstrate real-world value of task-specific models and modular LLM pipelines in enterprise settings

---

## 🧪 Part 1: Model Comparison – Fine-Tuned BERT vs LLaMA

We compared three models on a multi-label classification task using 10K labeled complaints:

| Model                  | Parameters | Accuracy | F1 Score | Precision | Recall |
|------------------------|------------|----------|----------|-----------|--------|
| BERT (base, fine-tuned)        | ~110M     | 73.19%   | 68.92%   | 70.00%    | 69.00% |
| LLaMA-1B (fine-tuned)          | ~1B       | 73.68%   | 72.58%   | 73.20%    | 74.30% |
| LLaMA-8B (zero-shot, base)     | 8B        | 14.76%   | 13.31%   | 32.10%    | 14.76% |

### 🧠 Key Insights

- Fine-tuned **LLaMA-1B** significantly outperforms the **8B base LLaMA**, despite being 8× smaller.
- Fine-tuned **BERT-base** performs nearly on par with LLaMA-1B — showing the strength of smaller, well-optimized models.
- Zero-shot large LLMs **fail without task adaptation**, underscoring the value of fine-tuning.

### 🛠️ Tools Used

- `transformers` (Hugging Face) for model fine-tuning
- `PEFT` and `bitsandbytes` for LoRA + quantization
- `scikit-learn` for metrics and evaluation
- `PyTorch` for training pipelines

---

## ✉️ Complaint Classification + Email Agent

This subsystem pairs the classifier with a lightweight LLM to **draft customer-facing replies**:

- Combined LoRA-tuned BERT with TinyLLaMA‑1.1B to label complaints and generate email responses
- Achieved 84% F1 in issue classification across 10K samples
- Added sentiment analysis using DistilBERT to personalize tone
- Used prompt engineering to ensure brand-aligned, context-aware email templates

This model forms the **input layer** for downstream triage and response automation.

---

## 🤖 Part 2: Agentic Complaint Triage (LangChain + FAISS + Gemini Flash)

This system performs **real-time triage** of customer complaints, designed for internal bank ops.

### 🔁 Workflow:

1. Embeds 5,000+ historical complaints using MiniLM (GPU-accelerated)
2. Indexes embeddings with FAISS for semantic retrieval
3. Wraps each complaint in a LangChain `Document` with metadata
4. Uses Gemini Flash with a structured prompt to output:
   - Issue category
   - Urgency score (0–10)
   - Responsible team (e.g., Deposits, Fraud)
   - Root cause hypothesis
   - Recommended resolution
   - Escalation flag (true/false)
   - Tags

### 🧾 Example Output

```json
{
  "issue_category": "Overdraft Fees",
  "urgency_score": 7,
  "team": "Retail Deposits Operations",
  "root_cause": "Paycheck posted after overdraft cutoff time",
  "suggested_resolution": "Refund fees or clarify policy",
  "escalate": false,
  "tags": ["overdraft", "fees", "cutoff", "refund"]
}
