# Documentation Guide for MorphoMapping PySide6 GUI

## 📋 Overview

This guide explains how to write and maintain documentation for the MorphoMapping PySide6 GUI. Follow this structure to ensure consistency and completeness.

## 📁 Documentation Structure

### Main Documentation Files

1. **`README_PYSIDE6.md`** - Quick start and technical overview
2. **`PYSIDE6_DEPLOYMENT.md`** - Installation and deployment instructions
3. **`../gui_user_guide.md`** - Complete user guide (step-by-step workflows)
4. **`../gui_installation.md`** - Installation guide for non-programmers

## ✍️ Documentation Writing Guidelines

### 1. User Guide (`gui_user_guide.md`)

**Target Audience:** End users (scientists, researchers)

**Structure:**
```markdown
# MorphoMapping GUI User Guide

## Introduction
- What is MorphoMapping GUI
- What can it do
- System requirements

## Getting Started
- Installation
- First launch
- Basic workflow overview

## Step-by-Step Workflows

### 1. Project Setup
- Setting Run-ID
- Project directory structure

### 2. Loading Data
- Selecting DAF files
- Drag & drop functionality
- File conversion process

### 3. Metadata Management
- Manual entry vs. upload
- Required columns (sample_id, group, replicate)
- Saving metadata

### 4. Feature Selection
- Include/exclude features
- Feature chips interface
- Population selection

### 5. Dimensionality Reduction
- Method selection (DensMAP, UMAP, t-SNE)
- Parameter adjustment
- Running analysis

### 6. Visualization
- Color by options
- Axis limits adjustment
- Highlighting cells
- Export options

### 7. Clustering
- Algorithm selection (KMeans, GMM, HDBSCAN)
- Parameter configuration
- Elbow plot (KMeans)
- Cluster statistics

## Advanced Features
- Top 10 Features export
- Cluster statistics bar chart
- Plot export (PNG/PDF)

## Troubleshooting
- Common issues
- Error messages
- Performance tips
```

**Writing Style:**
- Use clear, simple language
- Include screenshots where helpful
- Provide step-by-step instructions
- Use numbered lists for sequences
- Use bullet points for options/features

### 2. Installation Guide (`gui_installation.md`)

**Target Audience:** Non-programmers, first-time users

**Structure:**
```markdown
# MorphoMapping GUI Installation Guide

## Prerequisites
- Python installation
- Required packages
- System requirements

## Installation Steps
1. Install Python
2. Install dependencies
3. Download MorphoMapping
4. Verify installation

## First Launch
- Starting the application
- Initial setup
- Verifying everything works

## Troubleshooting Installation
- Common installation errors
- Dependency issues
- Platform-specific notes
```

### 3. Technical README (`README_PYSIDE6.md`)

**Target Audience:** Developers, technical users

**Structure:**
```markdown
# MorphoMapping PySide6 GUI

## Overview
- Architecture
- Technology stack
- Key features

## Quick Start
```bash
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn
python app_pyside6.py
```

## Architecture
- Component structure
- Threading model
- Memory management

## Development
- Code structure
- Adding features
- Testing

## API Reference
- Key classes
- Important functions
- Configuration options
```

### 4. Deployment Guide (`PYSIDE6_DEPLOYMENT.md`)

**Target Audience:** System administrators, deployment teams

**Structure:**
```markdown
# PySide6 Deployment Guide

## Deployment Options
- Standalone executable (PyInstaller)
- Python package installation
- Docker container

## Build Process
- Creating executables
- Packaging dependencies
- Distribution

## Platform-Specific Notes
- macOS
- Windows
- Linux

## Configuration
- Environment variables
- Configuration files
- User preferences
```

## 📝 Documentation Standards

### Code Blocks

Always include language tags:
```python
# Python code example
from PySide6.QtWidgets import QApplication
```

```bash
# Shell command example
python app_pyside6.py
```

### Screenshots

- Use descriptive filenames: `gui-feature-selection.png`
- Include alt text: `![Feature Selection Interface](gui-feature-selection.png)`
- Place in `assets/` directory

### Links

- Use relative paths for internal links
- Use descriptive link text
- Verify links work after updates

### Version Information

Always include:
- GUI version
- Python version requirements
- Dependency versions
- Last updated date

## 🔄 Documentation Update Checklist

When adding new features, update:

- [ ] **User Guide** - Add workflow section
- [ ] **README** - Update feature list
- [ ] **Installation Guide** - If dependencies change
- [ ] **Deployment Guide** - If build process changes
- [ ] **Code Comments** - Docstrings for new functions

## 📋 Feature Documentation Template

When documenting a new feature:

```markdown
### Feature Name

**Purpose:** What this feature does

**Location:** Where to find it in the GUI

**How to Use:**
1. Step one
2. Step two
3. Step three

**Parameters:**
- Parameter 1: Description
- Parameter 2: Description

**Output:** What you get

**Tips:**
- Helpful tip 1
- Helpful tip 2

**Troubleshooting:**
- Common issue: Solution
```

## 🎯 Current Documentation Status

### ✅ Completed
- Basic README structure
- Framework migration documentation
- Testing documentation

### 📝 Needs Update
- User guide (add new PySide6 features)
- Installation guide (PySide6 specific)
- Deployment guide (executable creation)

### 🆕 To Create
- Complete user workflow guide
- Troubleshooting guide
- FAQ section

## 📚 Reference Documentation

### For Users
- [GUI User Guide](../gui_user_guide.md)
- [GUI Installation Guide](../gui_installation.md)

### For Developers
- [GUI Concept](../gui_concept.md)
- [Framework Migration](FRAMEWORK_MIGRATION.md)
- [Testing Guide](TESTING.md)

## 🔗 Related Documentation

- [Main Documentation README](../README.md)
- [Quickstart Guide](../morphomapping_quickstart.md)
- [Step-by-Step Guide](../morphomapping_step_by_step.md)

