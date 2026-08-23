# akash5100 blog

Personal research notes, deep learning architectures, and experiments. Built with Hugo + PaperMod, optimized for Obsidian and GitHub Pages.

## Local Development

Run the Hugo server via Docker:

```bash
docker run --rm -it -v "$PWD:/src" -p 1313:1313 hugomods/hugo:exts hugo server --bind 0.0.0.0 --baseURL http://localhost:1313/
```

Open [http://localhost:1313/](http://localhost:1313/) in your browser.

## Writing with Obsidian

1. Open this repository (or just `content/posts/`) in **Obsidian**.
2. Create a new `.md` file in `content/posts/` with the standard front matter:

```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD
tags:
  - "deeplearning"
math: true
ShowToc: true
---
```

3. Write your research notes using standard Markdown, LaTeX formulas (`$$ ... $$` or `$ ... $`), and images.

## Deployment

Deployments are 100% automated. Whenever you `git push origin main`:
1. GitHub Actions pulls any Git LFS images and fetches PaperMod.
2. Builds the site with Hugo in ~15 seconds.
3. Automatically publishes your new post live to [akash5100.github.io](https://akash5100.github.io).