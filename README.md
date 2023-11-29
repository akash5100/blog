# blog
install dependencies
```bash
bundle install
bundle exec jekyll serve
```
to build
```
bundle exec jekyll build
```
to run in docker:
```
docker pull jekyll/jekyll
docker run --rm -it -p 4000:4000 -v "$(pwd):/srv/jekyll" jekyll/jekyll bash

bundle install
bundle exec jekyll serve --host 0.0.0.0
```

template automater usage
```bash
./create.py -t 'some title' -c 'what category'
```
