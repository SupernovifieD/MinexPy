# MinexPy

MinexPy is a practical toolkit for geoscience researchers and students who are learning how to solve real geoscientific problems with Python—or who want a deeper, more structured understanding of which Python tools fit which geoscience tasks.

Even though modern AI tools can help you prototype quickly, it’s still hard for beginners to answer questions like:

- Which library should I use for this exact problem?
- What’s the “standard” workflow in geoscience when using Python?
- How do I discover reliable tools without getting lost in hundreds of packages?

MinexPy aims to reduce that confusion by providing curated, beginner-friendly building blocks and sensible combinations of widely-used libraries, so you can focus more on the science and less on setup and guesswork.

---

## Installation

Install with pip:

```bash
pip install minexpy
```

---

## How to use

MinexPy provides both a Python API and a command-line interface.

### Python API

Like most Python packages, you install it and import what you need:

```python
import minexpy.stats as mstats
from minexpy import StatisticalAnalyzer, describe
```

### Command-Line Interface

After installation, use the `minexpy` command for quick access to documentation and examples:

```bash
# Show help and available functions
minexpy

# Show practical code examples you can copy
minexpy demo

# Open documentation in browser
minexpy docs

# Show package information
minexpy info
```

Documentation and examples will expand over time as modules are added.

---

## Roadmap / TODO

Want to contribute? Here are some high-impact areas to work on:

- Mapping module for geological, geochemical, and geophysical data

- Interpolation methods commonly used in geosciences (e.g., minimum curvature, triangulation, and others)

- AOI extraction from large satellite imagery (selecting/cropping a region of interest from big scenes)

---

## Contributing

Contributions are welcome—especially examples, notebooks/markdown tutorials, and new modules that follow the project’s goals:

- beginner-friendly

- practical workflows

- clear and detailed documentation using Google's docstrings guidelines
