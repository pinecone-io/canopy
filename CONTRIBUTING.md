# Contributing to Canopy
Thank you for considering contributing to Canopy! We appreciate the time and effort you put
into making this project better. Following these guidelines will help streamline the process
and make it easier for both contributors and maintainers.


## Issues
If you encounter any [issues](https://github.com/pinecone-io/canopy/issues/new/choose) while using the project, please report them. 
Include a detailed description of the problem, steps to reproduce it, and your environment details.

For any question, please use the `Discussions` section rather than opening an issue. This helps keep the issue tracker 
focused on bugs and feature requests.


## Pull Requests

It is really simple to get started and create a pull request. Canopy is released regularly, so, you should see your
improvements release in a matter of days or weeks 🚀

Unless your change is trivial (typo, docs tweak etc.), please create an issue to discuss the change before
creating a pull request.

If you're looking for something to get your teeth into, check out the
["good first issue"](https://github.com/pinecone-io/canopy/issues?q=is:issue+is:open+label:%22good+first+issue%22)
label on GitHub.


### Prerequisites

You'll need the following prerequisites:

- Any Python version between **Python 3.8 and 3.11**
- [**Poetry**](https://github.com/python-poetry/poetry)
- **git**
- **make**

### Installation & Setup

Fork the repository on GitHub and clone your fork locally.

```bash
# Clone your fork and cd into the repo directory
git clone git@github.com:<your username>/canopy.git
cd canopy

# Install canopy, dependencies and dev dependencies
make install
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
After making your changes, make sure to add the relevant tests. If you are adding a new feature, add unit tests.
If you are fixing a bug, add a regression test. Please make an effort not to copy and paste other tests, but instead
try to use existing fixtures or create new ones. Also, prefer parametrizing the tests over duplication where possible.

### Run linting, static type checking and unit tests
Run the following to make sure everything is working as expected:

```bash
# Run unit tests
make test-unit

# Lint the code
make lint 

# Run static type checking
make static

# There are a few more sub-commands in Makefile like which you might want to use.
# You can run `make help` to see more options.
```

### Run system and integration tests
In order to fully test the changes you need to create a Pinecone and an OpenAI account, 
and set the following environment variables:

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

You might need to include more environment variables based on the components you are testing. Please refer to the
[README](https://github.com/pinecone-io/canopy/blob/main/README.md) for more details.

In order to run all tests including system and integration tests, run the following:

```bash
# Run all tests
make test

# Alternatively, you can run the tests individually
# make test-system
# make test-e2e
```

### Commit and push your changes

Commit your changes, push your branch to GitHub, and create a pull request.

Please follow the pull request template and fill in as much information as possible. Link to any relevant issues and include a description of your changes.



