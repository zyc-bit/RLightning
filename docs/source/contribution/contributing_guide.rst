Contributing Guide
==================

Welcome to RLightning. This guide covers the expected development workflow for contributions to the project.

Types of Contributions
----------------------

We welcome the following types of contributions:

**Bug Reports**

- Use GitHub Issues to report bugs
- Include a minimal reproducible example
- Specify your environment (Python version, OS, CUDA version if applicable)
- Describe expected vs. actual behavior

**Bug Fixes**

- Reference the related issue in your PR
- Keep changes focused and minimal
- Document how you validated the fix

**Documentation**

- Fix typos, clarify explanations, add examples
- Ensure code examples stay runnable
- Follow the existing documentation style

**New Features**

- Discuss major features in an issue before implementation
- Follow the existing architecture patterns
- Include documentation for new user-facing behavior

**Performance Improvements**

- Include benchmarks showing the improvement
- Call out any tradeoffs or assumptions

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.11 or higher
- Git with submodule support
- `uv <https://docs.astral.sh/uv/>`_ package manager (recommended)

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone --recursive https://github.com/DeepLink-org/RLightning.git
   cd RLightning

The ``--recursive`` flag ensures submodules (for example ``third_party/rsl_rl``) are initialized.

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

For core development:

.. code-block:: bash

   uv sync --extra dev

For environment-specific development, add the corresponding extra:

.. code-block:: bash

   # ALE/Atari environments
   uv sync --extra dev --extra ale

   # MuJoCo environments
   uv sync --extra dev --extra mujoco

   # Isaac Lab environments
   uv sync --extra dev --extra isaaclab

   # ManiSkill environments
   uv sync --extra dev --extra maniskill

.. note::

   Some extras have conflicts (for example ``isaaclab`` conflicts with ``mujoco`` and ``maniskill``). Use separate virtual environments for conflicting extras.

Virtual Environment Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project uses isolated virtual environments in ``.venvs/`` for different dependency sets:

.. code-block:: bash

   make install-core
   make install-ale
   make install-mujoco
   make install-isaaclab
   make install-maniskill
   make install-rslrl

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run python -c "import rlightning"

Code Style
----------

Formatting
~~~~~~~~~~

We use `isort <https://pycqa.github.io/isort/>`_ with the Black profile for import sorting:

.. code-block:: bash

   # Sort imports
   uv run isort rlightning/ examples/

   # Check without modifying
   uv run isort --check-only rlightning/ examples/

The isort configuration is in ``pyproject.toml``:

.. code-block:: toml

   [tool.isort]
   profile = "black"
   known_first_party = ["rlightning"]

Type Hints
~~~~~~~~~~

- Add type hints to all public functions and methods
- Use ``typing`` module for complex types
- Prefer built-in generics (``list[int]`` over ``List[int]``) for Python 3.11+

Example:

.. code-block:: python

   def process_batch(
       observations: torch.Tensor,
       actions: torch.Tensor,
       rewards: list[float],
   ) -> dict[str, torch.Tensor]:
       """Process a batch of transitions."""
       ...

Docstrings
~~~~~~~~~~

Use Google-style docstrings for all public APIs:

.. code-block:: python

   def compute_returns(
       rewards: torch.Tensor,
       dones: torch.Tensor,
       gamma: float = 0.99,
   ) -> torch.Tensor:
       """Compute discounted returns from rewards.

       Args:
           rewards: Tensor of shape (T, N) containing rewards.
           dones: Tensor of shape (T, N) containing episode termination flags.
           gamma: Discount factor. Defaults to 0.99.

       Returns:
           Tensor of shape (T, N) containing discounted returns.

       Raises:
           ValueError: If rewards and dones have different shapes.
       """
       ...

General Guidelines
~~~~~~~~~~~~~~~~~~

- Keep lines under 100 characters
- Use descriptive variable names
- Prefer composition over inheritance
- Avoid global state and singletons
- Use ``pathlib.Path`` over string paths
- Keep documentation in sync with behavioral changes

Git Workflow
------------

Branch Naming
~~~~~~~~~~~~~

Use the following prefixes for branch names:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Prefix
     - Purpose
     - Example
   * - ``feat/``
     - New features
     - ``feat/async-rollout``
   * - ``fix/``
     - Bug fixes
     - ``fix/buffer-overflow``
   * - ``refactor/``
     - Code refactoring
     - ``refactor/engine-simplify``
   * - ``docs/``
     - Documentation changes
     - ``docs/api-reference``
   * - ``chore/``
     - Maintenance tasks
     - ``chore/dependency-refresh``

Branching Model
~~~~~~~~~~~~~~~

- ``develop``: Main development branch. All feature branches merge here.
- ``main``/``master``: Production-ready code. Only release merges.

Commit Messages
~~~~~~~~~~~~~~~

Write clear, descriptive commit messages:

.. code-block:: text

   <type>: <subject>

   <body>

Types:

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation only
- ``refactor``: Code refactoring
- ``chore``: Maintenance tasks

Submitting Changes
------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. Sync with upstream:

   .. code-block:: bash

      git fetch origin
      git rebase origin/develop

2. Run the relevant command path for your change:

   .. code-block:: bash

      uv run python -c "import rlightning"

3. Check code style:

   .. code-block:: bash

      uv run isort --check-only rlightning/ examples/

4. Update documentation if needed:

   .. code-block:: bash

      cd docs
      ../.venv-docs/bin/sphinx-build -b html source build/html -W

Creating a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~

1. Push your branch:

   .. code-block:: bash

      git push origin feat/your-feature

2. Create a PR targeting the ``develop`` branch

3. Fill in the PR template:

   - **Summary**: Brief description of changes
   - **Related Issue**: Link to related issue (if any)
   - **Validation**: How the changes were checked
   - **Checklist**: Confirm docs updated and notes are accurate

PR Review Process
~~~~~~~~~~~~~~~~~

1. **Maintainer review**: Changes are reviewed for correctness and style
2. **Validation review**: Reviewers may ask for runtime or documentation validation details
3. **Address feedback**: Push additional commits to address comments
4. **Approval**: Once approved, maintainers will merge

.. note::

   Keep PRs focused and reasonably sized. Large PRs are harder to review and more likely to have conflicts.

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs

   # Using the docs virtual environment
   ../.venv-docs/bin/sphinx-build -b html source build/html

   # Preview locally
   cd build/html
   python -m http.server 8000
   # Visit http://localhost:8000

Documentation Style
~~~~~~~~~~~~~~~~~~~

- Use reStructuredText (``.rst``) format
- Include code examples with ``.. code-block::``
- Add cross-references with ``:ref:`` and ``:doc:``
- Document all public APIs with docstrings

API Documentation
~~~~~~~~~~~~~~~~~

When source code changes affect public APIs, regenerate API documentation:

.. code-block:: bash

   cd docs
   ../.venv-docs/bin/sphinx-apidoc -o source/api/_generated ../rlightning        --force --separate --module-first -e

Getting Help
------------

If you have questions or need help:

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing docs for answers
