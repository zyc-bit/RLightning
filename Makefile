# RLightning Development Makefile

############## Virtual Environments ##############
VENV_ROOT := $(PWD)/.venvs

VENV_CORE = $(VENV_ROOT)/core
VENV_DEV = $(VENV_ROOT)/dev
VENV_ALE = $(VENV_ROOT)/ale
VENV_MUJOCO = $(VENV_ROOT)/mujoco
VENV_ISAACLAB = $(VENV_ROOT)/isaaclab
VENV_MANISKILL = $(VENV_ROOT)/maniskill

## algorithm-specific venv that contains algorithms used in examples
VENV_RSLRL_PYPROJECT = examples/wbc_tracking/pyproject.toml
VENV_RSLRL = $(VENV_ROOT)/rslrl

############## Makefile Commands ##############

.PHONY: help install-core install-dev install-isaaclab install-ale install-maniskill install-mujoco install-rslrl clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

############## Install Commands ##############
install-core: ## Install core development dependencies plus ALE and MuJoCo extras
	@echo "Setting up core environment..."
	@if [ ! -d "$(VENV_CORE)" ]; then uv venv $(VENV_CORE); else echo "Using existing $(VENV_CORE)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_CORE) uv sync --extra dev --extra ale --extra mujoco
	@echo "Core environment setup complete. To activate, run: source $(VENV_CORE)/bin/activate"

install-dev: ## Install dev dependencies
	@echo "Setting up dev environment..."
	@if [ ! -d "$(VENV_DEV)" ]; then uv venv $(VENV_DEV); else echo "Using existing $(VENV_DEV)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_DEV) uv sync --extra dev
	@echo "Dev environment setup complete. To activate, run: source $(VENV_DEV)/bin/activate"

install-isaaclab: ## Install IsaacLab dependencies
	@echo "Setting up IsaacLab environment..."
	@if [ ! -d "$(VENV_ISAACLAB)" ]; then uv venv $(VENV_ISAACLAB); else echo "Using existing $(VENV_ISAACLAB)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_ISAACLAB) uv sync --extra dev --extra isaaclab
	@echo "IsaacLab environment setup complete. To activate, run: source $(VENV_ISAACLAB)/bin/activate"

install-ale: ## Install ALE dependencies
	@echo "Setting up ALE environment..."
	@if [ ! -d "$(VENV_ALE)" ]; then uv venv $(VENV_ALE); else echo "Using existing $(VENV_ALE)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_ALE) uv sync --extra dev --extra ale
	@echo "ALE environment setup complete. To activate, run: source $(VENV_ALE)/bin/activate"

install-maniskill: ## Install ManiSkill dependencies
	@echo "Setting up ManiSkill environment..."
	@if [ ! -d "$(VENV_MANISKILL)" ]; then uv venv $(VENV_MANISKILL); else echo "Using existing $(VENV_MANISKILL)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_MANISKILL) uv sync --extra dev --extra maniskill
	@echo "ManiSkill environment setup complete. To activate, run: source $(VENV_MANISKILL)/bin/activate"

install-mujoco: ## Install MuJoCo dependencies
	@echo "Setting up MuJoCo environment..."
	@if [ ! -d "$(VENV_MUJOCO)" ]; then uv venv $(VENV_MUJOCO); else echo "Using existing $(VENV_MUJOCO)"; fi
	@UV_PROJECT_ENVIRONMENT=$(VENV_MUJOCO) uv sync --extra dev --extra mujoco
	@echo "MuJoCo environment setup complete. To activate, run: source $(VENV_MUJOCO)/bin/activate"

install-rslrl: ## Install RSL-RL dependencies
	@echo "Setting up RSL-RL environment..."
	@if [ ! -d "$(VENV_RSLRL)" ]; then uv venv $(VENV_RSLRL); else echo "Using existing $(VENV_RSLRL)"; fi
	@VIRTUAL_ENV=$(VENV_RSLRL) uv pip install "setuptools<70" wheel
	@VIRTUAL_ENV=$(VENV_RSLRL) uv pip install --no-build-isolation flatdict==4.0.1
	@UV_PROJECT_ENVIRONMENT=$(VENV_RSLRL) uv sync --project $(VENV_RSLRL_PYPROJECT)
	@echo "RSL-RL environment setup complete. To activate, run: source $(VENV_RSLRL)/bin/activate"

############### Clean Commands ##############
clean: ## Clean Python cache artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
