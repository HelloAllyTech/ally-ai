#!/usr/bin/env python3
"""
Test runner for Lifeline AI utility functions.
"""
import subprocess
import sys


def run_tests():
    """Run all utility function tests."""
    print("🧪 Running Lifeline AI utility function tests...")
    print("=" * 60)

    try:
        # Run pytest for all test files in utils directory
        subprocess.run(
            ["poetry", "run", "pytest", "app/tests/utils/", "-v", "--tb=short"],
            check=True,
        )

        print("\n✅ All tests passed!")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ Poetry not found. Please install poetry first.")
        print(
            "   Install with: curl -sSL https://install.python-poetry.org | python3 -"
        )
        return 1


def run_with_coverage():
    """Run tests with coverage report."""
    print("🧪 Running tests with coverage...")
    print("=" * 60)

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

        print("\n✅ All tests passed with coverage!")
        print("📊 Coverage report generated in htmlcov/")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ Poetry not found. Please install poetry first.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "coverage":
            sys.exit(run_with_coverage())
        else:
            print("Usage: python run_tests.py [coverage]")
            print("  no args: run all utility tests")
            print("  coverage: run tests with coverage report")
            sys.exit(1)
    else:
        sys.exit(run_tests())
