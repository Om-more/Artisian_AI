# AI-Powered Artisan Assistance Platform

A multimodal AI system designed to assist Indian artisans with product understanding, naming, knowledge guidance, and early-stage market positioning by combining image understanding, retrieval-augmented generation (RAG), and controlled LLM reasoning.

---

## Project Overview

Traditional artisans often struggle with presenting their products effectively in modern digital marketplaces. This project addresses that gap by building a grounded, explainable AI assistant that can:

- Understand artisan products from images
- Infer craft categories deterministically
- Answer domain-specific questions using curated knowledge sources
- Suggest product names and descriptions without hallucination
- Provide event awareness based on city/state (non-scraped)
- Lay the foundation for pricing guidance using marketplace references

The system avoids black-box predictions and focuses on controlled reasoning and transparency.

---

## System Architecture

User Input (Text / Image)  
→ BLIP (Image Captioning)  
→ Rule-based Category Inference (4 craft types)  
→ Intent Detection  
→ RAG (FAISS + Sentence Transformers)  
→ llama-3.2-(3B) via Ollama  
→ Grounded Response  


---

## 🛠️ Tech Stack

**Models**
- llama-3.2-3B (via Ollama)
- BLIP (Salesforce)

**Retrieval & NLP**
- Sentence-Transformers (all-MiniLM-L6-v2)
- FAISS
- PyPDF

**Backend**
- Python
**Data Sources**
- Curated PDFs / TXT files (craft theory, marketing, history)
- Structured JSON (national/state/city-level events)

---

## 🎨 Supported Craft Categories

- Ceramics  
- Pottery  
- Sculptures  
- Paintings  

Category inference is rule-based to ensure deterministic behavior.

---

## Key Features

### Multimodal Product Understanding
- Accepts image input
- Generates visual description using BLIP
- Infers craft category via keyword-based logic

### Retrieval-Augmented Knowledge (RAG)
- Indexes 10+ curated PDFs/TXT files
- Uses FAISS for semantic retrieval
- Ensures responses are grounded in provided knowledge

### Intent-Aware Prompt Control
- Separates creative naming, taxonomy, and knowledge queries
- Prevents irrelevant or over-generalized responses

### Hallucination Control
- Injects dynamic product grounding into prompts
- Constrains LLM outputs to inferred product context

### Event Awareness (Non-Scraped)
- Uses structured JSON for handicraft and art events
- Filters by city, state, and category
- Redirects users to official sources
