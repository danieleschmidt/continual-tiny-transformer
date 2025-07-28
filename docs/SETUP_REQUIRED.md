# Manual Setup Requirements

## Repository Administrator Tasks

The following items require manual setup by users with repository admin access:

### 1. GitHub Actions Workflows

Create workflow files in `.github/workflows/`:
- `ci.yml` - Continuous integration testing
- `release.yml` - Automated releases  
- `dependabot.yml` - Dependency updates

**Reference**: [GitHub Actions Starter Workflows](https://github.com/actions/starter-workflows)

### 2. Branch Protection Rules

Configure for `main` branch:
- Require PR reviews (1+ reviewers)
- Require status checks to pass
- Require branches to be up to date

### 3. Repository Settings

- Enable vulnerability alerts in Security tab
- Configure merge button options
- Set repository description and topics
- Add homepage URL if applicable

### 4. Pre-commit Hooks Setup

After cloning, developers should run:
```bash
pip install pre-commit
pre-commit install
```

### 5. Development Dependencies

Install development tools if not in requirements:
```bash
pip install black ruff pytest coverage
```

## Why Manual Setup?

These items require elevated permissions or affect repository-wide settings that automated tools cannot modify safely.