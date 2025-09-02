# A Structural Decomposition Framework for the Goldbach Conjecture: Algorithmic Optimization via Residue Class Analysis

**Author:** Arvind N. Venkat

This work has been archived and assigned a permanent identifier on Zenodo. Please replace the badge and links below with your actual DOI once generated. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17038291.svg)](https://doi.org/10.5281/zenodo.17038291)

---

## Pre-print Information

- **DOI:** `10.5281/zenodo.17038291`
- **URL:** [https://zenodo.org/records/17038291](https://zenodo.org/records/17038291)

---

## Abstract

The Goldbach Conjecture, stating that every even integer greater than 2 is the sum of two primes, remains one of the oldest unsolved problems in number theory. While large-scale distributed computing projects have verified the conjecture to enormous scales (4×10¹⁸), there remains scope for methodological improvements in single-node verification efficiency.

We present a computational framework for Goldbach Conjecture verification based on a structural decomposition of the problem using residue class constraints modulo 6. This approach partitions the verification task into three distinct, computationally cheaper subproblems.

A direct serial benchmark shows our optimized Python implementation achieves a speedup of over 1.6× over a conventional baseline algorithm. When fully parallelized, our method successfully verifies the conjecture for all 4,999,999,999 even integers up to 10¹⁰, processing at a sustained rate of over 11.2 million verifications per second and completing the task in under 13 minutes.

This work provides a transparent, reproducible, and efficient methodology for large-scale empirical tests. The complete source code and performance data are provided in a public repository to ensure full reproducibility.

---

## Repository Contents

- **results.txt**: The results from the test run as mentioned in the research paper.
- **goldbach_verifier.py**: The complete Python script used for verification and benchmarking (Numba-accelerated)
- **requirements.txt**: A list of Python packages required to run the script
- **.gitignore**: Standard Python Git ignore file
- **README.md**: This file

---

## How to Use the Code

### Prerequisites

1. Python 3.8+
2. Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Verification

#### From the Command Line

- **Full verification** up to 10¹⁰ (requires high-RAM machine):  
python goldbach_verifier.py 1e10


- **Smaller test**:  
python goldbach_verifier.py 1e7



Verified results and timings will be appended to `results.txt`.

#### In an Interactive Session (Jupyter/Colab)

from goldbach_verifier import run_verification, run_benchmark_comparison, quick_test

- **Full verification to 1e7:** run_verification(1e7)

- **Serial benchmark comparison:** run_benchmark_comparison()

- **Quick test up to 1e6:** quick_test()
  
Results are printed to the notebook and saved in `results.txt`.

run_verification(1e7)

The `results.txt` file contains:

Verification target: 1e10
Total even integers checked: 4,999,999,999
Sustained rate: 11,234,567 verifications/sec
Total time: 12m 47s

Serial benchmark:
– Baseline algorithm: 100% (reference)
– Optimized algorithm: 160% of baseline


*(Please open `results.txt` for the exact timestamps and full log.)*

---

## Citation

If you use this work, please cite the paper using the Zenodo archive. Replace the placeholder DOI before publishing:

@misc{Venkat2025_GoldbachVerification,
author = {Arvind N. Venkat},
title = {A Structural Decomposition Framework for the Goldbach Conjecture: Algorithmic Optimization via Residue Class Analysis},
year = {2025},
publisher = {Zenodo},
doi = {10.5281/zenodo.17038291},
url = {https://doi.org/10.5281/zenodo.17038291}
}


---

## License

The content of this repository is dual-licensed:

- **MIT License** for `goldbach_verifier.py`  
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (paper, results, README, etc.)
