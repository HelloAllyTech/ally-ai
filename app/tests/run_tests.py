#!/usr/bin/env python3
"""
Test runner for Lifeline AI utility functions.
"""
import subprocess
import sys

from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_tests():
    """Run all utility function tests."""
    logger.info("🧪 Running Lifeline AI utility function tests...")
    logger.info("=" * 60)

    try:
        # Run pytest for all test files in utils directory
        subprocess.run(
            ["poetry", "run", "pytest", "app/tests/utils/", "-v", "--tb=short"],
            check=True,
        )

        logger.info("\n✅ All tests passed!")
        return 0

    except subprocess.CalledProcessError as e:
        logger.info(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        logger.info("❌ Poetry not found. Please install poetry first.")
        logger.info(
            "   Install with: curl -sSL https://install.python-poetry.org | python3 -"
        )
        return 1


def run_with_coverage():
    """Run tests with coverage report."""
    logger.info("🧪 Running tests with coverage...")
    logger.info("=" * 60)

    try:
        # Run pytest with coverage
        subprocess.run(
            [
                "poetry",
                "run",
                "pytest",
                "app/tests/utils/",
                "--cov=app",
                "--cov-report=html",
                "--cov-report=term-missing",
                "-v",
            ],
            check=True,
        )

        logger.info("\n✅ All tests passed with coverage!")
        logger.info("📊 Coverage report generated in htmlcov/")
        return 0

    except subprocess.CalledProcessError as e:
        logger.info(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        logger.info("❌ Poetry not found. Please install poetry first.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "coverage":
            sys.exit(run_with_coverage())
        else:
            logger.info("Usage: python run_tests.py [coverage]")
            logger.info("  no args: run all utility tests")
            logger.info("  coverage: run tests with coverage report")
            sys.exit(1)
    else:
        sys.exit(run_tests())
