# Testing Setup for Lifeline AI

## 🎯 Overview

Comprehensive testing setup for utility functions in the Lifeline AI application. All tests are organized in a clean, maintainable structure within the `app/tests/` directory.

## 📁 Structure

```
app/tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── run_tests.py             # Test runner script
├── pytest.ini              # Pytest configuration
├── TESTING_SETUP.md         # Complete documentation
└── utils/                   # Modular utility tests
    ├── __init__.py
    ├── test_affirmation_counter.py                    # Affirmation counter tests (15+ tests)
    ├── test_client_positivity_lift_calculator.py      # Positivity lift tests (15+ tests)
    ├── test_reflective_listening_calculator.py        # Reflective listening tests (15+ tests)
    ├── test_common.py                                 # Common utility tests
    ├── test_counselor_interruption_calculator.py      # Interruption calculator tests
    ├── test_language_detector.py                      # Language detector tests
    ├── test_logger.py                                 # Logger utility tests
    ├── test_rate_limiter.py                           # Rate limiter tests
    ├── test_silence_calculator.py                     # Silence calculator tests
    ├── test_startup.py                                # Startup utility tests
    ├── test_structured_model_converter.py             # Model converter tests
    └── test_utterance_duration_calculator.py          # Duration calculator tests
```

## 🧪 Test Coverage

### Utility Functions Tested

1. **AffirmationCounter** (`test_affirmation_counter.py`)
   - Empty list handling
   - No affirmations present
   - Affirmations detection
   - Client vs counselor message filtering
   - Fuzzy matching capabilities
   - Mixed content scenarios
   - Case insensitivity and punctuation handling

2. **ClientPositivityLiftCalculator** (`test_client_positivity_lift_calculator.py`)
   - Empty data handling
   - Insufficient message count (< 10)
   - Sufficient message count (≥ 10)
   - Positive sentiment trends
   - Mixed role conversations
   - Edge cases and error scenarios
   - Fluctuating sentiment patterns

3. **ReflectiveListeningCalculator** (`test_reflective_listening_calculator.py`)
   - Empty data handling
   - Missing client/counselor messages
   - Embedding service integration
   - Error handling
   - Short message filtering
   - Mock service testing
   - High/low similarity scenarios

4. **Other Utilities** (placeholder tests ready)
   - Common utilities
   - Counselor interruption calculator
   - Language detector
   - Logger utilities
   - Rate limiter
   - Silence calculator
   - Startup utilities
   - Structured model converter
   - Utterance duration calculator

## 🚀 Quick Start

```bash
# 1. Install dependencies
poetry install --with dev

# 2. Run all utility tests
python app/tests/run_tests.py

# 3. Run with coverage
python app/tests/run_tests.py coverage
```

### Direct pytest Usage
```bash
# Run all utility tests
poetry run pytest app/tests/utils/ -v

# Run specific test file
poetry run pytest app/tests/utils/test_affirmation_counter.py -v

# Run tests for specific utility
poetry run pytest app/tests/utils/test_*_calculator.py -v

# Run with coverage
poetry run pytest app/tests/utils/ --cov=app --cov-report=html
```

## 🔧 Configuration

### Environment Variables
Tests run with a complete test environment configuration that includes:
- Mock API keys and credentials
- Test database settings (Weaviate, AWS)
- Debug logging level
- Isolated test data

### Pytest Configuration
- **Location**: `app/tests/pytest.ini`
- **Async Support**: Enabled with `asyncio_mode = auto`
- **Verbose Output**: Enabled by default
- **Test Discovery**: Automatically finds `test_*.py` files

## 📊 Test Categories

### Basic Tests (`test_utils.py`)
- **Purpose**: Core functionality validation
- **Coverage**: Happy path scenarios
- **Fixtures**: Uses shared test data
- **Count**: 15+ test cases

## 🎭 Fixtures Available

### Shared Fixtures (`conftest.py`)
- `sample_chat_messages`: Realistic conversation data
- `sample_affirmation_messages`: Messages with affirmations
- `sample_client_messages`: Client-only messages for sentiment analysis

### Test-Specific Fixtures
- Mock embedding services
- Custom message datasets
- Error simulation scenarios

## ✅ Benefits

1. **Clean Organization**: Tests are properly structured and easy to find
2. **Comprehensive Coverage**: Both happy path and edge cases covered
3. **Easy to Run**: Simple commands for different test scenarios
4. **Maintainable**: Clear naming conventions and documentation
5. **Isolated**: Tests don't interfere with each other or production code
6. **Fast**: Quick execution with proper mocking

## 🔄 Adding New Tests

1. **Create new test file**: `test_<module_name>.py`
2. **Import the module**: `from app.utils.module import function`
3. **Use existing fixtures**: Or create new ones in `conftest.py`
4. **Follow naming**: `test_<function_name>_<scenario>`
5. **Add docstrings**: Explain what each test validates

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Make sure environment variables are set in `conftest.py`
2. **Async Tests**: Use `@pytest.mark.asyncio` for async functions
3. **Mock Issues**: Ensure mocks are properly configured for async functions
4. **Dependencies**: Run `poetry install --with dev` to install test dependencies

### Debug Mode
```bash
# Run with debug output
poetry run pytest app/tests/ -v -s --tb=long

# Run single test
poetry run pytest app/tests/test_utils.py::TestAffirmationCounter::test_count_affirmations_empty_list -v
```

## 📈 Next Steps

1. **Add Integration Tests**: Test utility functions with real services
2. **Performance Tests**: Add benchmarks for utility functions
3. **Property-Based Tests**: Use hypothesis for more comprehensive testing
4. **CI/CD Integration**: Add tests to your build pipeline
5. **Coverage Goals**: Aim for 90%+ coverage on utility functions

---

**Happy Testing! 🎉**
