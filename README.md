# 3T-RAG-RRA-Screener 🔍

**Towards Reliable and Trustworthy Environmental Intelligence: Automated Scientific Literature Screening with LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official code for the research paper:

> **Towards Reliable and Trustworthy Environmental Intelligence: Automated Dataset Construction with Large Language Models**  
> Runzhi Wang, Xinlu Lin, Baoyi Zhong, Yumeng Zhao, Ruixing Huang, Yu Huang, Chotiwat Jantarakasem, Yang Xiang, Jun Ma  
> *Nature Sustainability*, 202X

## ✨ Overview

`3T-RAG-RRA-Screener` is a hierarchical, multi-agent AI framework designed to automate the labor-intensive process of screening scientific literature for environmental research. It addresses the critical bottleneck in building large-scale environmental datasets by combining:

- **3T+RAG Prompting**: Domain-tailored prompting (Target-Tech-Type) grounded in Retrieval-Augmented Generation (RAG) to reduce hallucinations and improve accuracy.
- **Multi-Agent Review (RRA)**: A Reviewer-Reviewer-Arbiter architecture that enables traceable, auditable decision-making through deliberative discussions and conflict resolution.
- **High-fficiency & Cost-Effective**: Achieves expert-level accuracy at 1/20th the cost and 78× faster than manual screening.

## 📦 Installation

### 1. Clone the Repository
### 2. Create and Activate a Virtual Environment
### 3. Install Dependencies
### 4. Configure API Keys
### Expected Output
The script will process each article through the 3T+RAG-RRA pipeline and save results to the specified xlsx file, including:

- `article_id`: Unique identifier
- `title`: Article title
- `relevance_label`: Final label (Relevant/Irrelevant)
- `reviewer1_decision`, `reviewer2_decision`: Individual reviewer judgments
- `arbiter_decision`: Decision of arbiter agent if reviewer agents disagreed
- `final_decision`: Finial decision of RRA framework
- `reasoning_trace`: Step-by-step reasoning for auditability
## 📊 Usage

### 1. Prepare Your Data
### 2. Run the Full Pipeline
### 3. Evaluate Performance
