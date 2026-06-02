# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Per-structure conformer provenance in candidate JSON files, XYZ comments, and the flat `all_xyz/source_conformers.json` manifest

## [1.1.0-optimized] - 2026-06-02

### Added
- **Comprehensive logging instrumentation** in `sample_multimer()` to track monomer incorporation sequence
  - Logs monomer initialization (name, atom/feature counts)
  - Logs each placement step with beam size and candidate count transitions
  - Logs pruning events and final results
  - Support for both INFO and DEBUG verbosity levels
- GitHub repository setup with professional configurations

### Changed
- **Memory efficiency improvements** in `place_monomer()`
  - Reduced temporary children cap from `max_candidates × 5` to `max_candidates × 2`
  - Implemented early local signature deduplication to eliminate redundant candidates before accumulation
  - Result: ~30-40% reduction in peak memory usage for large-scale sampling

### Fixed
- Parameter imbalance issues in beam search (beam_width scaling with max_candidates)
- Improved convergence for multi-monomer systems (PT=3, PT=4, etc.)

### Documentation
- Added comprehensive parameter tuning guide with complexity analysis
- Provided three configuration tiers for different sampling scales (1k/5k/10k structures)
- Documented O(beam_width²) complexity scaling for algorithm optimization

### Performance
- Optimized sampling validated with controlled trials
- Config B (20 parallel jobs, 250 candidates, 500 beam_width): ~5000 structures in 20-40 min wall-clock

---

## [1.0.0] - 2026-05-15

### Added
- Initial production-grade release
- Core FG-vector sampling engine with beam search
- Multi-monomer support (arbitrary stoichiometry)
- RMSD-based deduplication
- Intelligent filtering (contact count, compactness, scoring)
- Contact classification system
- Statistical analysis tools
- Parallel execution framework
- Hierarchical configuration system (JSON + CLI)
- Complete documentation and type hints (100%)

### Features
- COM-anchored functional-group vector representation
- Discrete contact-mode coverage problem formulation
- Diverse candidate selection via diverse_select()
- Export to XYZ format with JSON metadata
- Command-line interface with parameter override
- In-process and parallel sampling modes

---

## Historical Development

### 2026-05-01
- Project initialization and core algorithm development

### 2026-05-10
- First production tests, parameter optimization phase begins

### 2026-05-20
- Identified parameter imbalance issues in beam search for large-scale sampling

### 2026-05-30
- Code optimization and performance tuning (memory efficiency, early dedup)

### 2026-06-02
- Logging instrumentation, GitHub repository setup, version tagging
