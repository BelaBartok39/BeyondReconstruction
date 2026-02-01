# Contributing to RF Anomaly Detection

Thank you for your interest in contributing to this research project!

## Ways to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include reproduction steps, expected behavior, and system info
- For dataset issues, specify the data source and format

### Code Contributions

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature`
3. **Make changes** following the code style below
4. **Run tests**: `pytest tests/ -v`
5. **Submit a PR** with a clear description

### Code Style

- Use Python 3.9+
- Follow PEP 8 with these preferences:
  - Double quotes for strings
  - Type hints for function signatures
  - Docstrings for public functions (Google style)
- Use `pathlib.Path` over `os.path`
- Run `black` and `isort` before committing

### Testing

- Add tests for new functionality in `tests/`
- Ensure existing tests pass: `pytest tests/ -v`
- Target 80%+ coverage for new code

### Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for developer guidance
- Add docstrings to new functions

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CLP_Project.git
cd CLP_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black isort

# Run tests
pytest tests/ -v
```

## Areas of Interest

We especially welcome contributions in:

- **New anomaly types**: Implementations in `src/data/synthetic.py`
- **Detection methods**: New detectors in `src/detection/`
- **Real-world validation**: Results on new RF datasets
- **Architecture improvements**: VAE variants in `src/models/`

## Research Collaboration

If you're interested in research collaboration (joint publications, dataset sharing), please contact the authors directly.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
