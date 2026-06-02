name: Pull Request
description: Create a pull request to contribute changes

body:
  - type: markdown
    attributes:
      value: |
        ## Pull Request Checklist
        Please ensure your PR meets the following requirements:
        
        - [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
        - [ ] Code follows PEP 8 style and passes `black`, `flake8`, `mypy`
        - [ ] All tests pass: `pytest tests/ -v --cov=fg_vector_sampler`
        - [ ] Documentation is updated (docstrings, README, or CHANGELOG)
        - [ ] Type hints are included (100% coverage)
        - [ ] Commits follow Conventional Commits format
        
  - type: textarea
    id: description
    attributes:
      label: Description
      description: What does this PR change and why?
    validations:
      required: true
      
  - type: textarea
    id: related_issues
    attributes:
      label: Related Issues
      description: "Link to related issues: Closes #123, Fixes #456"
      placeholder: "Closes #123"
      
  - type: textarea
    id: performance
    attributes:
      label: Performance & Memory Impact
      description: Document any performance, memory, or sampling quality changes
      placeholder: |
        - Memory: ~10% reduction in peak usage for large beams
        - Time: <1% overhead for small jobs
        - Quality: No impact on candidate diversity
        
  - type: textarea
    id: testing
    attributes:
      label: Testing
      description: Describe how you tested these changes
      placeholder: |
        - Ran controlled trial with max_candidates=250, beam_width=500
        - Verified 50 candidates returned with no regressions
        - Checked logging output shows monomer sequence correctly
      
  - type: checkboxes
    id: pr_type
    attributes:
      label: PR Type
      options:
        - label: "🐛 Bug Fix"
        - label: "✨ New Feature"
        - label: "⚡ Performance Improvement"
        - label: "📚 Documentation"
        - label: "♻️ Refactoring"
        - label: "🔧 Configuration/Build"
