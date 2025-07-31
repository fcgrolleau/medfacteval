# MedFactEval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-PSB%202025-b31b1b)](https://placeholder-link-to-paper.com)

Official implementation for the paper: **"MedFactEval and MedAgentBrief: A Framework and Pipeline for Generating and Evaluating Factual Clinical Summaries"**.

This repository provides the complete source code for **MedFactEval**, a framework for the scalable, fact-grounded evaluation of AI-generated clinical text, and **MedAgentBrief**, a model-agnostic workflow for generating high-quality, factual discharge summaries.

## Overview

Evaluating the factual accuracy of LLM-generated clinical text is a critical barrier to adoption. Manual expert review is the gold standard but is unscalable for continuous quality assurance. MedFactEval addresses this by reframing the ambiguous task of "is this a good summary?" into a series of concrete, verifiable questions based on clinician-defined key facts, which are then assessed by a robust LLM Jury.

For full details on the methodology and experimental results, please see our paper:
**[Link to your arXiv preprint or final publication]**

## Key Features

*   **MedFactEval Framework**: A scalable evaluation pipeline that uses an LLM Jury to assess the factual presence and consistency of AI-generated text against clinician-defined key facts.
*   **MedAgentBrief Workflow**: A model-agnostic, multi-step pipeline for generating high-fidelity clinical summaries from unstructured notes.
*   **Benchmark Data**: A benchmark for discharge summary evaluation based on the public MIMIC dataset to facilitate further research.
*   **Analysis Code**: Scripts to replicate the meta-evaluation, non-inferiority analysis, and figures presented in our paper.

## Repository Structure

```bash
medfacteval/
├── medfacteval/          # Core code for the MedFactEval framework and LLM Jury
├── medagentbrief/        # Code for the MedAgentBrief generation workflow
├── samples/                 # Anonymized samples of a MedAgentBrief summary and a MedFactEval evaluation repo 
├── requirements.txt      # Project dependencies
└── README.md
```
