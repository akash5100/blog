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

* Build Docker image
  ```bash
  make build
  ```

* Start the server (need for first time to install `gem` dependencies)
  ```bash
  make create
  ```

* get container name using:
  ```bash
  docker ps -a (IMAGE: blog)
  ```

* Everytime, to run server locally, do this:
  ```bash
  make run CONTAINER_NAME=relaxed_mcclintock 
  ```
  replace `relaxed_mcclintock` with generated container name

* Remove the Docker image (if needed)
  ```bash
  make clean
  ```

## Template Automater

```bash
./create.py -t 'some title' -c 'what category'  # Need chmod +x
```