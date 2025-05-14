# English Sentence Analyzer

![Project Banner](https://via.placeholder.com/800x200.png?text=English+Sentence+Analyzer)

A comprehensive tool for analyzing English sentences, featuring:
- **CEFR Level Classification** (A1-C2)
- **Tense Classification** (Present/Past/Future)
- Multiple interface options (GUI, Web, CLI)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features ✨

### CEFR Classifier
- Predicts language proficiency level (A1 to C2)
- Supports multiple ML models (SVM, Random Forest, Neural Network)
- Includes probability distribution across levels
- Visualizations (level distribution, word clouds)

### Tense Classifier
- Identifies verb tenses (Present, Past, Future)
- Uses Bidirectional LSTM model
- Combines model predictions with grammatical indicators
- Confidence scores for each tense

### Interfaces
- Desktop GUI (Tkinter)
- Web Interface (PyWebIO)
- Command Line Interface

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/English-Sentence-Analyzer.git
cd English-Sentence-Analyzer
