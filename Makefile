.PHONY: create clean build run

create:
	docker run -it -p 4000:4000 -v "$(PWD):/srv/jekyll" blog sh -c "bundle install && bundle exec jekyll serve --host 0.0.0.0"

clean:
	docker rmi -f blog

build:
	docker build -t blog --build-arg BUNDLE_INSTALL=true .

run:
	docker container start $(CONTAINER_NAME)

stop:
	docker container stop $(CONTAINER_NAME)
