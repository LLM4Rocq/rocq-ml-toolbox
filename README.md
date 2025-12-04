# rocq-ml-toolbox
A toolbox providing Rocq environment generation, an inference server, and project-parsing tools for ML-oriented interaction with the Rocq prover.

---

## Overview

This repository brings together several components to simplify interaction with the Rocq proof assistant, especially for machine-learning workflows:

* **Docker environment creator**
  Builds reproducible Rocq setups.

* **Inference server**
  Provides an API for sending commands to Rocq prover at scale. (overlay, based on Rocq-lsp/pétanque)

* **Project parser**
  Parses Rocq projects and extracts structured datasets. (based on previous works [crrrocq](), [llm4docq](), [goal2tac]()).

Each component can be used independently or combined to form a pipeline for ML-based experimentation.

---

## Repository Structure

```
rocq-ml-toolbox/
├── rocq_docker/        # Build and manage Rocq environments
├── inference_server/   # Inference server
├── client/             # Client library for interacting with the server
└── scraper/            # Tools for parsing Rocq projects
```

---

## Getting Started

### Requirements

* Docker (for environment generation)
* Python 3.9+ (for server, client, and scraper)

### Quick Start

TODO

---

## Goals

* Make Rocq more accessible for ML experimentation.
* Provide reproducible and modular tooling.
* Enable efficient project scraping and dataset generation.