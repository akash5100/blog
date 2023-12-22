# Blog

## Installation

```bash
bundle install          # Install dependencies
bundle exec jekyll serve # Run the server
```

To build:

```bash
bundle exec jekyll build # Build the project
```

## Docker

```bash
make build  # Build Docker image
make start  # Start the server
make clean  # Remove the Docker image
```

## Template Automater

```bash
./create.py -t 'some title' -c 'what category'  # Need chmod +x
```