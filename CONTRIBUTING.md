# Contributing to Canopy
Thank you for considering contributing to Canopy! We appreciate the time and effort you put
into making this project better. Following these guidelines will help streamline the process
and make it easier for both contributors and maintainers.


## Issues
If you encounter any [issues](https://github.com/pinecone-io/canopy/issues/new/choose) while using the project, please report them. 
Include a detailed description of the problem, steps to reproduce it, and your environment details.

For any question, please use the `Discussions` section rather than opening an issue. This helps keep the issue tracker 
focused on bugs and feature requests.

## Feature requests
If you have a feature request, please open an issue and describe the feature you would like to see, using the "Feature request" template.

## Contributing code

It is really simple to get started and create a pull request. Canopy is released regularly, so, you should see your
improvements released in a matter of days or weeks 🚀

If this is your first contribution to Canopy, you can start by looking at issues with the
["good first issue"](https://github.com/pinecone-io/canopy/issues?q=is:issue+is:open+label:%22good+first+issue%22)
label on GitHub.  
If you find an issue that you'd like to work on, please assign the issue to yourself and leave a comment to let others know that you are working on it. Feel free to start a discussion on the issue to discuss optional designs or approaches.

### Building from source
If you are planning to contribute to Canopy, you will need to create your own fork of the repository. 
If you just want to test the code locally, you can clone the repository directly.

1. Fork the repository on GitHub and clone your fork locally.

    ```bash
    # Clone your fork and cd into the repo directory
    git clone git@github.com:<your username>/canopy.git
    cd canopy
    ```
2. Install poetry, which is required for dependency management. It is recommended to install poetry in a virtual environment.
    You can install poetry using pip
    ```bash
   pip install poetry
    ```
   or using the following command

    ```bash
    # Install poetry
    curl -sSL https://install.python-poetry.org | python3 -
    ```
3. Install the dependencies and dev dependencies
    ```bash
    # Install canopy, dependencies and dev dependencies
   poetry install --with dev
    ```
4. Set up accounts and define environment variables 
   Please refer to the [README](./README.md#mandatory-environment-variables) for more details.
5. Remember to activate the virtual environment before running any commands
    ```bash
    # Activate the virtual environment
    poetry shell
    ```
   or alternatively, you can run the commands directly using `poetry run`
    ```bash
    # Run the command inside the virtual environment
    poetry run <command>
    ```
#### Optional - installing extra dependencies
Canopy has a few optional dependencies, mostly for additional service providers. If you want to use Canopy with any of these providers, please make sure to install the relevant extra. For example, to use Canopy with Cohere, you should install with:
 ```bash
 # Install canopy, with the cohere extra
 poetry install --with dev --extras cohere
  ```

### Running tests
Canopy uses unit tests, system tests and end-to-end tests. Unit tests verify the functionality of each code module, without any external dependencies. System tests verify integration with services like Pinecone DB and OpenAI API. End-to-End tests verify the functionality of the entire Canopy server.     
System and end-to-end tests require valid API keys for Pinecone and Open AI. Some optional providers require additional environment variables, and are otherwise skipped.  
You can create a single `.env` file in the root directory of the repository and set all the environment variables there.

To run all tests, run the following command:
```bash
# Run all tests
poetry run pytest tests/
```
You can also run only one type of tests using the following commands:
```bash
# Run unit tests
poetry run pytest tests/unit

# Run system tests
poetry run pytest tests/system

# Run end-to-end tests
poetry run pytest tests/e2e
```

### Check out a new branch and make your changes
Create a new branch for your changes.

```bash
# Checkout a new branch and make your changes
git checkout -b my-new-feature-branch
# Make your changes...
```

### Document your changes
When contributing to Canopy, please make sure that all code is well documented. 

The following should be documented using properly formatted docstrings:

- Modules
- Class definitions
- Function definitions
- Module-level variables

Canopy uses [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) formatted 
according to [PEP 257](https://www.python.org/dev/peps/pep-0257/) guidelines. 
(See [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 
for further examples.)

[pydoclint](https://github.com/jsh9/pydoclint) is used for linting docstrings. You can run `make lint` to check your docstrings.

If you are making changes to the public API, please update the documentation in the README.md file.

### Add the relevant tests
All code changes to Canopy need to be covered by tests. After making your changes, make sure to add relevant unit tests. 
Tests that require external integration (e.g. communication with an API or service) should be placed under the `tests/system/` directory.  

Please make an effort to avoid code duplication. Some unit tests have a common base class that can be extended. Other tests use fixtures to parameterize test cases over several subclasses. Instead of copy-pasting other test cases, try to utilize these mechanisms as much as possible.

### Run linting, static type checking and unit tests
Run the following to make sure everything is working as expected:

```bash
# Run unit tests
make test-unit
# If you don't have make installed, you can run the following command instead
poetry run pytest tests/unit

# Lint the code
make lint 
# Or alternatively
poetry run flake8 .

# Run static type checking
make static
# Or
poetry run mypy src
```
(There are a few more sub-commands in Makefile like which you might want to use. You can run `make help` to see more options.)

### Commit your changes, push to GitHub, and open a Pull Request

Commit your changes, push your branch to GitHub, the use GitHub's website to create a pull request.

Please follow the pull request template and fill in as much information as possible. Link to any relevant issues and include a description of your changes.



