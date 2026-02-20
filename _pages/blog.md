---
permalink: /blog/
title: "Blog"
excerpt: ""
author_profile: true
---

<div class="blog-hub">
  <section class="blog-hub__hero">
    <p class="blog-hub__eyebrow">Notes & Updates</p>
    <h1 class="blog-hub__title">Blog</h1>
    <p class="blog-hub__subtitle">
      Thoughts on research, systems, and practical engineering.
    </p>
  </section>

  {% assign featured = site.posts | first %}
  {% if featured %}
    <section class="blog-hub__section">
      <div class="blog-hub__section-head">
        <h2>Featured</h2>
      </div>

      <article class="blog-card blog-card--featured" itemscope itemtype="http://schema.org/CreativeWork">
        <a class="blog-card__link" href="{{ featured.url | relative_url }}" aria-label="{{ featured.title }}"></a>
        <div class="blog-card__content">
          <p class="blog-card__meta">
            <time datetime="{{ featured.date | date_to_xmlschema }}">{{ featured.date | date: "%b %d, %Y" }}</time>
          </p>
          <h3 class="blog-card__title" itemprop="headline">{{ featured.title }}</h3>
          {% if featured.excerpt %}
            <p class="blog-card__excerpt">{{ featured.excerpt | strip_html | strip_newlines | truncate: 220 }}</p>
          {% endif %}
          {% if featured.tags and featured.tags.size > 0 %}
            <p class="blog-card__tags">
              {% for tag in featured.tags limit:4 %}
                <span class="blog-tag">{{ tag }}</span>
              {% endfor %}
            </p>
          {% endif %}
        </div>
      </article>
    </section>

    {% assign post_count = site.posts | size %}
    {% if post_count > 1 %}
      <section class="blog-hub__section">
        <div class="blog-hub__section-head">
          <h2>All Posts</h2>
        </div>

        <div class="blog-grid">
          {% for post in site.posts offset:1 %}
            <article class="blog-card" itemscope itemtype="http://schema.org/CreativeWork">
              <a class="blog-card__link" href="{{ post.url | relative_url }}" aria-label="{{ post.title }}"></a>
              <div class="blog-card__content">
                <p class="blog-card__meta">
                  <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %d, %Y" }}</time>
                </p>
                <h3 class="blog-card__title" itemprop="headline">{{ post.title }}</h3>
                {% if post.excerpt %}
                  <p class="blog-card__excerpt">{{ post.excerpt | strip_html | strip_newlines | truncate: 150 }}</p>
                {% endif %}
                {% if post.tags and post.tags.size > 0 %}
                  <p class="blog-card__tags">
                    {% for tag in post.tags limit:3 %}
                      <span class="blog-tag">{{ tag }}</span>
                    {% endfor %}
                  </p>
                {% endif %}
              </div>
            </article>
          {% endfor %}
        </div>
      </section>
    {% endif %}
  {% else %}
    <section class="blog-hub__empty">
      <h2>No posts yet</h2>
      <p>New articles will appear here soon.</p>
    </section>
  {% endif %}
</div>
