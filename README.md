# Bank Consumer Complaints Analysis: Comparing LLaMA and BERT Models
This project analyzes a dataset of bank consumer complaints to evaluate the performance of different language models on a classification task. The goal is to explore how fine-tuning smaller models compares to using large, general-purpose models.

## Models Compared:
BERT (bert-base-uncased): A smaller, well-optimized model with ~300 million parameters, fine-tuned for this dataset.
Fine-tuned LLaMA-1B (meta-llama/Llama-3.2-1B-Instruct): A smaller variant of LLaMA with 1 billion parameters, fine-tuned for better task-specific performance.
Base LLaMA-8B (meta-llama/Llama-3.1-8B): A large-scale model with 8 billion parameters, evaluated without fine-tuning.
Key Findings:
Fine-tuned LLaMA-1B and BERT models significantly outperform the base LLaMA-8B model.
Despite its smaller size (~300 million parameters), BERT achieves results comparable to the fine-tuned LLaMA-1B model.
The base LLaMA-8B model performs poorly, emphasizing the importance of fine-tuning over sheer model size.
Results:
Model	Parameters	Accuracy	F1 Score	Precision	Recall
BERT (bert-base-uncased)	~110 million	73.19%	68.92%	70.00%	69.00%
Fine-tuned LLaMA-1B	~1 billion	73.68%	72.58%	73.20%	74.30%
Base LLaMA-8B	8 billion	14.76%	13.31%	32.10%	14.76%
Note: Fine-tuned smaller models (BERT and LLaMA-1B) achieve significantly better performance than the base LLaMA-8B model, which fails to generalize effectively without fine-tuning.

## Objectives:
Explore the efficiency of fine-tuning smaller models for domain-specific tasks.
Highlight the diminishing returns of increasing model size without optimization.
Demonstrate the cost-effectiveness and practicality of smaller, task-specific models compared to large, expensive models.
Tools and Frameworks:
Transformers library for working with LLaMA and BERT models.
PyTorch for model training and evaluation.
Scikit-learn for metrics and confusion matrix visualization.
Hugging Face Datasets for dataset preprocessing and tokenization.


## Dataset: https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp/data

The project uses a bank consumer complaints dataset, with preprocessing to extract relevant features (narrative) and labels (product). The data is split into training, validation, and test sets, and performance is evaluated using standard classification metrics like accuracy, F1 score, precision, and recall.


## Conclusion:
This project demonstrates that fine-tuning smaller models like BERT and LLaMA-1B can outperform or match much larger models (e.g., LLaMA-8B) on domain-specific tasks. This finding underscores the value of efficient, targeted fine-tuning in real-world applications, both for performance and computational cost.
