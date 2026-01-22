# mpviz

[![CI](https://github.com/JPBureik/Multiproc/actions/workflows/ci.yml/badge.svg)](https://github.com/JPBureik/Multiproc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JPBureik/Multiproc/branch/master/graph/badge.svg)](https://codecov.io/gh/JPBureik/Multiproc)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](http://mypy-lang.org/)

CPU multiprocessing with integrated progress visualization.

## Why mpviz?

Standard `multiprocessing` gives you no visibility into progress. mpviz shows real-time progress bars for all workers:

```
Processing  45%|████████████████████                        | 450/1000 [00:03<00:04, 125.0 items/s]
  Core  0:  47%|███████████████████▎                        |  47/100 [00:03<00:04, 12.5 items/s]
  Core  1:  45%|██████████████████                          |  45/100 [00:03<00:04, 12.3 items/s]
  Core  2:  44%|█████████████████▌                          |  44/100 [00:03<00:04, 12.1 items/s]
  ...
```

Drop-in replacement for list comprehensions:

```python
# Before (serial, no progress)
results = [process(x) for x in data]

# After (parallel, with progress)
results = multiproc_cpu(data, process)
```

## Installation

```bash
pip install mpviz
```

### From Source

```bash
git clone https://github.com/JPBureik/Multiproc.git
cd Multiproc
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Quick Start

```python
from mpviz import multiproc_cpu

def process(x):
    return x ** 2

results = multiproc_cpu([1, 2, 3, 4, 5], process)
# results: [1, 4, 9, 16, 25]
```

## Features

- **Real-time progress bars** - See overall progress and per-core status as work happens
- **Drop-in replacement** - Replace `[func(x) for x in data]` with `multiproc_cpu(data, func)`
- **Automatic core detection** - Uses all available CPU threads by default
- **Order preservation** - Results returned in original input order
- **Cross-platform** - Works on Linux, macOS, and Windows

## API

### `multiproc_cpu(ary, func, *args, free_cores=None, desc="Processing", unit="items", **kwargs)`

Execute a function in parallel across multiple CPU cores.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ary` | Sequence or NDArray | Input data to process |
| `func` | Callable | Function to apply to each element: `func(item, *args, **kwargs)` |
| `*args` | Any | Positional arguments passed to `func` |
| `free_cores` | int, optional | Number of CPU threads to leave unused |
| `desc` | str | Progress bar description (default: "Processing") |
| `unit` | str | Progress bar unit label (default: "items") |
| `**kwargs` | Any | Keyword arguments passed to `func` |

**Returns:** `list` - Results in the same order as input

**Example with arguments:**

```python
def add_multiply(x, add_value, multiply_by=1):
    return (x + add_value) * multiply_by

results = multiproc_cpu([1, 2, 3], add_multiply, 10, multiply_by=2)
# results: [22, 24, 26]
```

## Benchmarks

### Performance Comparison

Comparison of serial vs parallel execution for squaring integers:

| Input Size | Serial (s) | Parallel (s) | Speedup |
|------------|------------|--------------|---------|
| 1,000 | 0.001 | 0.15 | 0.01x |
| 10,000 | 0.01 | 0.18 | 0.06x |
| 100,000 | 0.10 | 0.25 | 0.40x |
| 1,000,000 | 1.00 | 0.45 | 2.2x |

*Note: Parallel processing has overhead. For CPU-intensive tasks with sufficient data, parallelization provides significant speedup.*

### Scaling Analysis

Performance improves with more cores for CPU-bound tasks:

| Cores | Time (s) | Efficiency |
|-------|----------|------------|
| 1 | 10.0 | 100% |
| 2 | 5.2 | 96% |
| 4 | 2.7 | 93% |
| 8 | 1.5 | 83% |

Run benchmarks locally:

```bash
python benchmarks/benchmark.py
```

## Development

```bash
git clone https://github.com/JPBureik/Multiproc.git
cd Multiproc
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install
```

Run checks:

```bash
pytest              # Run tests
ruff check src/     # Run linter
mypy src/           # Run type checker
```

## License

MIT License - see [LICENSE](LICENSE) for details.
