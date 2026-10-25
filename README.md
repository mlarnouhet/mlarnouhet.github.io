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

Edit `site.config.mjs` to set your name, biography, headline, and homepage profile links. The initial biography describes the supplied notes; it does not claim an affiliation or credentials. An empty author name uses “Machine learning notes” as the site title.

The Home page includes a profile photo to the left of the introduction (above it on small screens). To change your photo, put it in `public/images/` and update `profileImage.src` in `site.config.mjs` to its path, such as `/images/profile.jpg`, along with descriptive `profileImage.alt` text. Photos are cropped to a square.

Replace `public/CV.pdf` with your CV, keeping the same filename and capitalization. The homepage's CV link opens the PDF in a new tab for inline viewing, without requesting a download.

The initial article date is the date this site was prepared, 16 October 2026. Change the `date` in its frontmatter if it should use a different publication date.

## Add a post

Create `src/content/blog/your-post.md`:

```md
---
title: "Your article title"
description: "A short summary for the blog index."
date: 2026-10-16
---

Your article starts here. The template supplies its title, so start sections at `##`.

Inline math: $\nabla_\theta \mathcal{L}$.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$
```

Fenced code blocks get syntax highlighting. Public images go in `public/images/`; images next to a post can also use relative Markdown paths. All Markdown posts in `src/content/blog/` are included in the website. Articles with more than two second-level headings get a contents list automatically.

Astro 7 uses its official Unified Markdown processor to support the plan's `remark-math` and `rehype-katex` plugins. KaTeX fonts and CSS are bundled locally, and the site uses system fonts.

## Static hosting

The output works on Cloudflare Pages or GitHub Pages. Configure `SITE_URL` as the full production origin before building to generate the correct canonical and sharing URLs. It overrides `site.config.mjs`.

For Cloudflare Pages, use `npm run build` and output directory `dist`. If the repository contains this `website` subdirectory, use `website` as the project root.

For GitHub Pages hosted beneath a repository path, also set `BASE_PATH=/repository-name` at build time. All internal navigation and bundled assets respect this path. Upload `dist` using the hosting provider's static deployment workflow.

The `.openai/hosting.json` file holds the identity and static-output settings for the separate private Sites deployment. It is not required by other static hosts.
