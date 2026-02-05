## Running Tests with Pytest and Generating Coverage Reports

This project uses [pytest](https://docs.pytest.org/) for running tests and [pytest-cov](https://pytest-cov.readthedocs.io/) for generating coverage reports.

To run the tests and generate a coverage report in XML format, use the following command:

```bash
python -m pytest --cov=app --cov-report=xml
```