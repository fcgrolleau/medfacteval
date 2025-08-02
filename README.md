# MedFactEval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper](https://img.shields.io/badge/paper-under_review-b31b1b)

Official implementation for the paper:  **"MedFactEval and MedAgentBrief: A Framework and Pipeline for Generating and Evaluating Factual Clinical Summaries"**

This repository provides the complete source code for **MedFactEval**, a framework for scalable, fact-grounded evaluation of AI-generated clinical text, and **MedAgentBrief**, a model-agnostic workflow for generating high-quality, factual discharge summaries.

---

## Overview

Evaluating the factual accuracy of LLM-generated clinical text is a critical barrier to adoption. Manual expert review is the gold standard but is unscalable for continuous quality assurance.

**MedFactEval** reframes the ambiguous task of "is this a good summary?" into a series of concrete, verifiable questions based on clinician-defined key facts, which are then assessed by a robust LLM Jury for both factual presence and contradictions.

**MedAgentBrief** provides a multi-step, model-agnostic pipeline for generating high-fidelity clinical summaries from unstructured notes, using structured prompts and expert-informed templates.

---

## Key Features

- **MedFactEval Framework:**  
  Scalable evaluation pipeline using an LLM Jury to assess factual presence and contradictions of AI-generated text against clinician-defined key facts.

- **MedAgentBrief Workflow:**  
  Model-agnostic, multi-step pipeline for generating high-fidelity clinical summaries from unstructured notes, with clear formatting and medical relevance.

- **Single-Prompt Summarizer:**  
  An alternative, streamlined summarization approach using a single, comprehensive prompt for rapid prototyping or baseline comparison.

- **Analysis Code:**  
  Jupyter notebooks and scripts to replicate the meta-evaluation, non-inferiority analysis, and performance trade-offs presented in our paper.

- **Samples:**  
  Anonymized examples of both MedAgentBrief-generated summaries and MedFactEval evaluation reports for reference.

---

## Repository Structure

```bash
medfacteval/
├── medfacteval/                # Core code for the MedFactEval framework and LLM Jury
│   ├── auto_eval.py            # Main evaluation pipeline and reporting (LLM-as-a-judge approach)
│   ├── agnostic_evaluator_models.py # LLM and model-agnostic evaluation utilities
│   └── multi_evaluator.py      # Multi-model evaluation (LLM Jury approach)
│
├── medagentbrief/              # Code for the MedAgentBrief generation workflow
│   ├── medagentbrief.py        # Main multi-step summarization pipeline
│   ├── llm_calls.py            # LLM interaction utilities
│   └── html_summary_generator.py # HTML report generation (matching tags to citations)
│
├── single_prompt/              # Single-prompt summarization baseline
│   └── single_prompt_summarizer.py
│
├── analysis/                   # Meta-evaluation, statistical analysis, and figure generation
│   ├── agreements.ipynb        # Reproduces Figure 2
│   ├── non_inferiority.ipynb   # Reproduces Figure 3 and Figure S3
│   └── tradeoffs.ipynb         # Reproduces Figure 1 and Figure S1 and Figure S2
│
├── samples/                    # Anonymized samples of MedAgentBrief summaries and MedFactEval reports
│   ├── anonymized_medagentbrief_sumary.html
│   └── anonymized_medfacteval_report.html
│
├── supplementary_materials.pdf # Supplementary materials (Table S1, Figures S1-S3)
└── README.md
```

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medfacteval.git
   cd medfacteval
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secure LLM API Access:**  
   This project is designed to work with HIPAA-compliant LLM APIs (e.g., Stanford Health Care or Vertex AI) for handling patient health information.

   - **For MedAgentBrief** (`medagentbrief/llm_calls.py`):  
     - Set your API key by editing the line:  
       `lab_key = "insert_your_lab_key_or_load_it_from_env"`
     - Or provide your Google Cloud JSON key in `credentials_path`.

   - **For MedFactEval**:  
     - Provide your credentials in `medfacteval/agnostic_evaluator_models.py`.

   - For more details on secure API access, please see our documentation:  
     [Secure LLM API Access Guide](https://github.com/HealthRex/CDSS/blob/master/scripts/DevWorkshop/llm-api/phi-llm-api-python.md)

4. **Run MedAgentBrief or MedFactEval:**  
   See the docstrings in `medagentbrief/medagentbrief.py` and `medfacteval/multi_evaluator.py` for example usage and input formats.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaborations, please contact François Grolleau: [grolleau@stanford.edu](mailto:grolleau@stanford.edu)