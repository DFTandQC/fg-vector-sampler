# Contributing to FG-Vector Sampler

Thank you for your interest in contributing to FG-Vector Sampler! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/fg-vector-sampler.git
cd fg-vector-sampler
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

## Development Workflow

### Code Style

- **Python**: Follow [PEP 8](https://pep8.org/) with 100-char line limit
- **Type Hints**: 100% type hint coverage required
- **Formatting**: Use `black` with default settings
- **Linting**: Pass `flake8` and `mypy` checks

```bash
# Format code
black fg_vector_sampler/

# Check style
flake8 fg_vector_sampler/
mypy fg_vector_sampler/

# Run tests
pytest tests/ -v --cov=fg_vector_sampler
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples**:
- `feat(sampler): add early deduplication in place_monomer()`
- `fix(core): correct rotation matrix calculation for O atoms`
- `docs(readme): add parameter tuning guide`
- `perf(analysis): optimize contact graph computation`

### Testing

- Add tests for all new features
- Ensure all existing tests pass
- Aim for >90% code coverage

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fg_vector_sampler --cov-report=html
```

## Contribution Types

### Bug Reports
Create an issue with:
- Clear, descriptive title
- Minimal reproducible example
- Expected vs actual behavior
- Python/OS version and environment info

### Feature Requests
Include:
- Use case and motivation
- Proposed implementation (optional)
- Potential impact on performance/memory

### Documentation
- Fix typos and clarify confusing sections
- Add examples for complex features
- Update docstrings with better explanations

### Performance Improvements
- Benchmark before/after
- Document memory and time impacts
- Include profiling data if significant

## Pull Request Process

1. **Update Documentation**: Docstrings, README, or CHANGELOG as needed
2. **Run Checks**:
   ```bash
   black fg_vector_sampler/
   flake8 fg_vector_sampler/
   mypy fg_vector_sampler/
   pytest tests/ -v --cov=fg_vector_sampler
   ```
3. **Create Pull Request**:
   - Link related issues: `Closes #123`
   - Describe changes clearly
   - Add performance/memory impact if relevant

4. **Code Review**: Address feedback from maintainers

5. **Merge**: Once approved and tests pass

## Architecture Guidelines

### Module Organization
- `sampler.py`: Core beam search algorithm
- `core.py`: Geometry and data structures
- `features.py`: Functional group handling
- `priors.py`: Contact scoring and filtering
- `analysis.py`: Statistical analysis tools
- `io.py`: File I/O utilities
- `cli.py`: Command-line interface
- `molecule_lib.py`: Molecule template management

### Key Constraints
- Beam search complexity is O(beam_width²) per monomer placement
- Memory peaks with temporary children accumulation
- Maintain COM-anchoring for reproducibility

### Performance Considerations
- Early deduplication critical for large-scale sampling
- Consider beam_width ≈ max_candidates × N_monomers rule
- Profile memory with large parameter values

## Reporting Issues

Use GitHub Issues with labels:
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Docs need clarification or update
- `performance`: Speed/memory optimization
- `question`: User inquiry

## Questions?

- Check [README.md](README.md) and [documentation](docs/)
- Review [existing issues](https://github.com/DFTandQC/fg-vector-sampler/issues)
- Start a [discussion](https://github.com/DFTandQC/fg-vector-sampler/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to FG-Vector Sampler! 🎉**
