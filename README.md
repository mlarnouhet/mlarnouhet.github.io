# Machine learning notes

A small, static Astro website following `../personal_ml_website_plan.md`.

- `/` is the About page.
- `/blog/` lists published articles, newest first.
- `/blog/diffusion-models/` contains the complete supplied notes, with KaTeX equations.
- The original Markdown files in the parent directory are unchanged.

## Run locally

Use Node.js 22.22 or newer (`nvm use` selects the version in `.nvmrc`).

```sh
npm ci
npm run dev
```

Open the local URL printed by Astro, normally `http://localhost:4321`.

```sh
npm run check
npm run build
npm run preview
```

The production website is generated in `dist/`. There is no backend or database.

## Personalize

Edit `site.config.mjs` to set your name, biography, headline, and profile links. The initial biography describes the supplied notes; it does not claim an affiliation or credentials. An empty author name uses “Machine learning notes” as the site title. Adding a GitHub profile also adds it to the top navigation.

The initial article date is the date this site was prepared, 16 October 2026. Change the `date` in its frontmatter if it should use a different publication date.

## Add a post

Create `src/content/blog/your-post.md`:

```md
---
title: "Your article title"
description: "A short summary for the blog index."
date: 2026-10-16
tags: [Machine learning]
draft: false
---

Your article starts here. The template supplies its title, so start sections at `##`.

Inline math: $\nabla_\theta \mathcal{L}$.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$
```

Fenced code blocks get syntax highlighting. Public images go in `public/images/`; images next to a post can also use relative Markdown paths. Set `draft: true` to exclude a post from the archive and generated routes. Articles with more than two second-level headings get a contents list automatically.

Astro 7 uses its official Unified Markdown processor to support the plan's `remark-math` and `rehype-katex` plugins. KaTeX fonts and CSS are bundled locally, and the site uses system fonts.

## Static hosting

The output works on Cloudflare Pages or GitHub Pages. Configure `SITE_URL` as the full production origin before building to generate the correct canonical and sharing URLs. It overrides `site.config.mjs`.

For Cloudflare Pages, use `npm run build` and output directory `dist`. If the repository contains this `website` subdirectory, use `website` as the project root.

For GitHub Pages hosted beneath a repository path, also set `BASE_PATH=/repository-name` at build time. All internal navigation and bundled assets respect this path. Upload `dist` using the hosting provider's static deployment workflow.

The `.openai/hosting.json` file holds the identity and static-output settings for the separate private Sites deployment. It is not required by other static hosts.
