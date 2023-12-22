FROM jekyll/jekyll:latest as builder
WORKDIR /srv/jekyll
COPY Gemfile Gemfile.lock ./
ARG BUNDLE_INSTALL
RUN if [ "$BUNDLE_INSTALL" = "true" ]; then \
      gem update --system && \
      gem install bundler; \
    fi
COPY . .

FROM jekyll/jekyll:latest
WORKDIR /srv/jekyll
COPY --from=builder /srv/jekyll /srv/jekyll
EXPOSE 4000
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]
