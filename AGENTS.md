# AI Agent Guidelines for MinexPy Development

This document provides standards and best practices for AI agents assisting with MinexPy development. Following these guidelines ensures consistency, quality, and maintainability across the codebase.

---

## Project Overview

**MinexPy** is a geoscience toolkit designed for researchers and students. It emphasizes:
- **Beginner-friendly APIs** with clear, practical workflows
- **NumPy-style documentation** with comprehensive examples
- **Standard scientific Python stack** (NumPy, SciPy, Pandas, Matplotlib)
- **Research notes** that explain the mathematical and scientific foundations

---

## Core Development Principles

### 1. Code Standards

**Python Best Practices:**
- Follow PEP 8 style guidelines
- Use type hints for all function signatures (see `minexpy/stats.py` for examples)
- Write clean, readable code with meaningful variable names
- Keep functions focused on a single responsibility

**Dependency Management:**
- **Primary dependencies:** NumPy, SciPy, Pandas, Matplotlib
- Only use these libraries unless the user explicitly requests alternatives
- If functionality is missing from these libraries, implement it from scratch with optimized logic (fast and memory-efficient)
- Avoid adding new dependencies without explicit user approval

### 2. Documentation Requirements

**NumPy Docstring Standard:**

All functions, classes, and methods must follow NumPy's docstring conventions with these sections:

```python
def example_function(data: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Brief one-line summary.
    
    Extended description explaining what the function does, why it's useful,
    and any important context (especially for geoscience applications).
    
    Parameters
    ----------
    data : array-like
        Description of the parameter, including acceptable types and formats.
        Mention NaN handling if applicable.
    threshold : float, default 0.5
        Description with default value clearly stated.
        
    Returns
    -------
    dict
        Description of return value, including keys if returning a dict,
        or shape/type if returning arrays.
        
    Examples
    --------
    >>> import numpy as np
    >>> from minexpy.module import example_function
    >>> 
    >>> data = np.array([1.2, 3.4, 5.6, 7.8])
    >>> result = example_function(data, threshold=0.6)
    >>> print(result)
    
    Notes
    -----
    Additional technical details, assumptions, or algorithmic information.
    Mention computational complexity if relevant.
    
    References
    ----------
    .. [1] Author. (Year). Title. Journal, Volume(Issue), pages.
    """
```

**Documentation Completeness:**
- Include at least one practical example in every docstring
- Explain parameters thoroughly, including valid ranges and default behaviors
- Document NaN handling explicitly (most functions should use `nan_policy='omit'`)
- Add "Notes" section for technical details or geoscience-specific context
- Include "References" for algorithms or methods from literature

### 3. Module Development Workflow

When creating or modifying code in the `minexpy/` directory:

**Step 1: Implement the Code**
- Write the function/class with full type hints
- Handle NaN values appropriately (typically exclude them)
- Add comprehensive NumPy-style docstrings with examples
- Support multiple input types: `np.ndarray`, `pd.Series`, `List[float]`

**Step 2: Update Documentation Files**
- Add or update relevant pages in `docs/user-guide/` for usage tutorials
- Add or update `docs/api/` for API reference (if not auto-generated)
- Update navigation in `mkdocs.yml` if adding new documentation pages

**Step 3: Add Research Notes (When Applicable)**
- Create or update files in `docs/research-notes/` to explain:
  - Mathematical foundations
  - Geoscience context and applications
  - Algorithm rationale and trade-offs
- **Important:** Write substantive, educational content—not generic AI-generated text
- Include equations (using LaTeX), diagrams, or references to help users understand the "why" behind the implementation

**Step 4: Update Package Exports**
- Add new functions/classes to `minexpy/__init__.py` for easy importing
- Update CLI help and demo text

### 4. Testing and Quality

**Before Finalizing Changes:**
- Test with various input types (arrays, Series, lists)
- Verify NaN handling works correctly
- Check that examples in docstrings actually run
- Ensure type hints are accurate
- Verify documentation builds correctly with MkDocs

### 5. Research Notes Guidelines

Research notes are a **key differentiator** for MinexPy.

**What to Include:**
- Clear explanations of mathematical concepts (with LaTeX equations, and do not write equations inline with text)
- Motivation: Why is this method used in geoscience?
- Worked examples with real-world geoscience context
- Comparisons between methods (when applicable)
- Limitations and assumptions
- References to primary literature

**What to Avoid:**
- Generic, surface-level explanations
- Copy-pasted Wikipedia content
- AI-generated text without domain expertise
- Missing equations or technical depth
- Lack of geoscience-specific context

**Organization:**
- Group related topics logically (e.g., "Statistical Foundations", "Mapping Basics")
- Use clear headings and subheadings
- Cross-reference with API documentation where appropriate

---

## Restrictions

### Configuration Files - DO NOT MODIFY Unless Explicitly Asked

**Strictly Protected Files:**
- `.github/workflows/*.yml` - CI/CD pipeline configurations
- `mkdocs.yml` - Only modify if explicitly asked (except adding new pages to `nav` section)
- `pyproject.toml` - Package configuration and dependencies
- `.gitignore` and other Git configuration files

**Rationale:** These files control critical infrastructure and should only be changed deliberately by the user.

### When Making Changes to Navigation

If adding new documentation pages, you **may** add entries to the `nav` section in `mkdocs.yml`, but:
- Follow the existing structure and indentation
- Place pages in the appropriate category
- Do not modify other parts of `mkdocs.yml` (theme, plugins, extensions, etc.)

---

## Version Control & Release Management

### Commit Message Standard

All commit messages **must follow the Angular commit convention** for consistency and automated changelog generation.

**Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types (required):**
- `feat`: New feature for the user
- `fix`: Bug fix
- `docs`: Documentation changes only
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates, build changes
- `ci`: Changes to CI/CD configuration

**Scope (optional but recommended):**
- Module or area affected: `stats`, `correlation`, `statviz`, `docs`, `ci`, etc.

**Subject:**
- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period at the end
- Maximum 50 characters

**Good Commit Message Examples:**

```
feat(stats): add geometric mean calculation function

Implements geometric mean with support for arrays, Series, and lists.
Includes comprehensive NumPy-style documentation and examples.

Closes #42
```

```
fix(correlation): handle edge case with constant arrays

Prevents division by zero when calculating correlation with arrays
that have zero variance. Now returns NaN with appropriate warning.
```

```
docs(research-notes): add correlation methods mathematical background

Explains Pearson vs Spearman correlation with equations, assumptions,
and geoscience application examples.
```

```
chore(deps): update numpy to 1.26.0
```

**Bad Examples (Don't Do This):**

```
❌ Updated stuff
❌ Fixed bug
❌ Add new feature.
❌ FEAT: Added correlation function
```

### Changelog Management

**When to Update CHANGELOG.md:**

Update `CHANGELOG.md` **whenever you bump the version** in `pyproject.toml`. This typically happens when:
- Preparing for a new release
- After significant features or fixes have accumulated
- User explicitly requests a version bump

**Changelog Format:**

Follow the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New features that have been added

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in upcoming releases

### Removed
- Features that have been removed

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes

## [0.2.0] - 2026-02-16

### Added
- Geometric mean calculation in stats module
- New research notes for correlation methods

### Fixed
- Division by zero in correlation with constant arrays
```

**Changelog Workflow:**

1. **During Development:** Add entries to the `[Unreleased]` section as you complete features/fixes
2. **Before Release:** 
   - Update version in `pyproject.toml`
   - Move `[Unreleased]` items to a new version section with date
   - Add comparison links at bottom if present
   - Create empty `[Unreleased]` section for future changes

**Example Entry Categories:**

- **Added:** New functions, classes, modules, features, documentation pages
- **Changed:** Modifications to existing functionality, API changes
- **Fixed:** Bug fixes, error handling improvements
- **Deprecated:** Features marked for future removal (rare in early versions)
- **Removed:** Deleted features or APIs
- **Security:** Security-related fixes (important to highlight)

---

## Code Examples and Patterns

### Example: Function with Full Documentation

See `minexpy/stats.py` functions like `skewness()`, `kurtosis()`, or `describe()` as reference implementations.

### Example: Class-Based Analyzer

See `StatisticalAnalyzer` class in `minexpy/stats.py` for the pattern:
- Support both single arrays and DataFrames
- Provide convenience methods for common operations
- Return appropriate types (float vs Series vs DataFrame)

### Example: Input Handling Pattern

```python
def function_name(data: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Function docstring here."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        raise ValueError("Input data is empty or contains only NaN values")
    
    # Your logic here
    return result
```

---

## Quick Reference Checklist

When developing new functionality:

- [ ] Code follows Python/PEP 8 standards
- [ ] Uses only approved dependencies (NumPy, SciPy, Pandas, Matplotlib)
- [ ] Includes full type hints
- [ ] Has NumPy-style docstring with all sections
- [ ] Includes at least one runnable example
- [ ] Handles NaN values appropriately
- [ ] Supports multiple input types (array, Series, list)
- [ ] Added to `minexpy/__init__.py` for easy importing
- [ ] User guide documentation added/updated in `docs/user-guide/`
- [ ] Research notes added/updated in `docs/research-notes/` (if applicable)
- [ ] Navigation updated in `mkdocs.yml` (only `nav` section if needed)
- [ ] Tested with various input types
- [ ] Documentation builds without errors
- [ ] Commit message follows Angular convention
- [ ] CHANGELOG.md updated (in `[Unreleased]` section or new version section if bumping version)

---

## Questions or Uncertainties?

When in doubt:
1. **Look at existing code** in `minexpy/stats.py` or `minexpy/correlation.py` as templates
2. **Ask the user** before adding new dependencies or making architectural decisions
3. **Prioritize clarity** over cleverness—code should be educational for beginners
4. **Document thoroughly**—assume users are learning both Python and geoscience

---

**Remember:** MinexPy is designed to help beginners learn. Every piece of code, documentation, and research note should contribute to that educational mission.