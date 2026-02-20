---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I am currently working on inference acceleration for text-to-image and text-to-video generation, with a particular focus on efficient Diffusion Transformer (DiT) inference.

If you are interested in academic collaboration, feel free to contact me at **yuanahao574@gmail.com**.

I am an undergraduate student at the College of Computer and Big Data, Fuzhou University. My research interests include machine learning systems, AI infrastructure, and text-to-image/text-to-video systems.


{% comment %}
# 🔥 News
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.
- *2022.02*: &nbsp;🎉🎉 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.

# 📝 Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2016</div><img src='{{ "/images/500x300.png" | relative_url }}' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Kaiming He**, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=DhtAFkwAAAAJ&citation_for_view=DhtAFkwAAAAJ:ALROH1vI_8AC) <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong>
- Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet.
</div>
</div>

- [Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet](https://github.com), A, B, C, **CVPR 2020**
{% endcomment %}

# 🎖 Honors and Awards
- *2025.10* National Scholarship (Undergraduate)

# 📖 Educations
- *2023.08 - Present*, Undergraduate, College of Computer and Big Data, Fuzhou University, Software Engineering.
- *2020.09 - 2023.06*, Shandong Experimental High School.

{% comment %}
# 💻 Internships
- *2019.05 - 2020.02*, [Lorem](https://github.com/), China.
{% endcomment %}

# ✍️ Blog

I occasionally write notes about research, engineering, and experiments.

{% assign recent_posts = site.posts | slice: 0, 3 %}
{% if recent_posts.size > 0 %}
{% for post in recent_posts %}
- *{{ post.date | date: "%Y.%m.%d" }}*: [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
{% else %}
- No posts yet. Check back soon.
{% endif %}

[View all posts]({{ "/blog/" | relative_url }})
