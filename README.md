# unitares-core

[![License: MIT](https://img.shields.io/badge/license-MIT--with--attribution-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

The thermodynamic dynamics engine for [UNITARES](https://github.com/CIRWEL/unitares) — the ODE, coherence, stability, and drift math that evolves each agent's EISV state vector.

This package is a build-time Cython-compiled dependency of the main `unitares` server. It can be installed and used standalone for research / reproduction of the paper's dynamics, but the governance decisions, dashboard, MCP surface, and knowledge graph live in the parent repo.

## Install

```bash
pip install "unitares-core @ git+https://github.com/CIRWEL/unitares-core.git@v2.3.0"
```

Requires Python 3.12+, NumPy, and a Cython-capable build toolchain (Xcode CLT on macOS; `build-essential` on Linux).

For local development against the source:

```bash
git clone https://github.com/CIRWEL/unitares-core.git
cd unitares-core
pip install -e .
```

## What's in here

Nine modules under `governance_core/`:

| Module | Purpose |
|--------|---------|
| `dynamics.py` | Coupled ODE for E (energy), I (integrity), S (entropy), V (valence) |
| `coherence.py` | Coherence metric C(V, Θ) |
| `stability.py` | Lyapunov-style stability analysis |
| `ethical_drift.py` | Four-component behavioral–epistemic drift vector (calibration / complexity / coherence / stability) + L2 norm |
| `phase_aware.py` | Phase-aware governance hooks |
| `adaptive_governor.py` | Adaptive parameter tuning |
| `parameters.py` | Default parameter sets + bounds |
| `scoring.py` | Risk / health scoring |
| `utils.py` | Shared helpers |

All four drift components and their coupling to the EISV dynamics are described in §Behavioral–Epistemic Drift Vector of the [UNITARES v6 paper](https://doi.org/10.5281/zenodo.19647159) — that paper is the authoritative reference; this repo is the reference implementation.

## Relationship to UNITARES

- **unitares-core** (this repo) — pure math, no I/O, no daemons, no databases. Cython-compiled for the production server; pure-Python importable for tests.
- **unitares** — governance MCP server, dashboard, knowledge graph, check-in pipeline, dialectic system. Imports from `governance_core`.

The split exists so the dynamics layer can be reasoned about, tested, and reproduced independently of the operational machinery.

## Paper citation

If you use this code in research, please cite the UNITARES v6 paper:

```bibtex
@misc{wang2026unitares,
  author       = {Wang, Kenny},
  title        = {UNITARES: Information-Theoretic Governance of Heterogeneous Agent Fleets},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19647159},
  url          = {https://doi.org/10.5281/zenodo.19647159}
}
```

Kenny Wang — ORCID: [0009-0006-7544-2374](https://orcid.org/0009-0006-7544-2374)

## License

MIT with attribution requirement — see [LICENSE](LICENSE).
