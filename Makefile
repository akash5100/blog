.PHONY: start clean build

start:
	docker run --rm -it -p 4000:4000 -v "$(PWD):/srv/jekyll" blog-iie sh -c "bundle install && bundle exec jekyll serve --host 0.0.0.0"


clean:
	docker rmi -f blog-iie

build:
	docker build -t blog-iie --build-arg BUNDLE_INSTALL=true .
