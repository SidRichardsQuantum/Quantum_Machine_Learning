from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_site"
SITE_URL = "https://SidRichardsQuantum.github.io/Quantum_Machine_Learning/"
REPO_URL = "https://github.com/SidRichardsQuantum/Quantum_Machine_Learning"
PYPI_URL = "https://pypi.org/project/qml-pennylane/"

DOCS = [
    ("Overview", ROOT / "README.md", "overview.html", "Package overview and quick-start examples."),
    (
        "Usage",
        ROOT / "USAGE.md",
        "usage.html",
        "API, CLI, benchmarking, and reproducibility guide.",
    ),
    ("Theory", ROOT / "THEORY.md", "theory.html", "Mathematical background for the workflows."),
    ("Changelog", ROOT / "CHANGELOG.md", "changelog.html", "Release notes and project history."),
    (
        "Variational Quantum Classifier",
        ROOT / "docs/qml/variational_quantum_classifier.md",
        "variational-quantum-classifier.html",
        "Binary classification with trainable parameterised quantum circuits.",
    ),
    (
        "Variational Regression",
        ROOT / "docs/qml/variational_regression.md",
        "variational-regression.html",
        "Regression workflows built from reusable PennyLane components.",
    ),
    (
        "Quantum CNN",
        ROOT / "docs/qml/qcnn.md",
        "qcnn.html",
        "Hierarchical quantum convolution and pooling for classification.",
    ),
    (
        "Quantum Autoencoder",
        ROOT / "docs/qml/autoencoder.md",
        "autoencoder.html",
        "Compression and reconstruction of structured quantum state families.",
    ),
    (
        "Quantum Kernels",
        ROOT / "docs/qml/quantum_kernels.md",
        "quantum-kernels.html",
        "Feature-map kernels and support-vector classification workflows.",
    ),
    (
        "Metric Learning",
        ROOT / "docs/qml/metric_learning.md",
        "metric-learning.html",
        "Trainable quantum embeddings supervised by contrastive losses.",
    ),
    (
        "Classical Baselines",
        ROOT / "docs/qml/classical_baselines.md",
        "classical-baselines.html",
        "Classical reference models for QML benchmark comparisons.",
    ),
    (
        "Benchmarks",
        ROOT / "docs/qml/benchmarks.md",
        "benchmarks.html",
        "Deterministic multi-seed comparisons across implemented models.",
    ),
]

ALGORITHMS = [
    (
        "Variational quantum classifier",
        "Train compact PennyLane classifiers on synthetic binary datasets.",
        "variational-quantum-classifier.html",
        ["VQC", "Classification", "PennyLane"],
    ),
    (
        "Variational quantum regression",
        "Fit continuous targets with a trainable quantum regressor.",
        "variational-regression.html",
        ["VQR", "Regression", "Optimization"],
    ),
    (
        "Quantum convolutional neural network",
        "Use shared quantum convolution blocks and pooling-style reductions.",
        "qcnn.html",
        ["QCNN", "Four qubits", "Classifier"],
    ),
    (
        "Quantum autoencoder",
        "Learn latent quantum representations and reconstruction maps.",
        "autoencoder.html",
        ["Autoencoder", "Compression", "Fidelity"],
    ),
    (
        "Quantum kernel methods",
        "Build quantum feature-map kernels for SVM workflows.",
        "quantum-kernels.html",
        ["Kernels", "SVM", "Feature maps"],
    ),
    (
        "Quantum metric learning",
        "Train embedding geometry with contrastive supervision.",
        "metric-learning.html",
        ["Metric learning", "Embeddings", "Contrastive"],
    ),
]


def slugify(value: str, separator: str = "-") -> str:
    return re.sub(r"[^a-z0-9]+", separator, value.lower()).strip(separator)


def render_markdown(path: Path) -> str:
    md = markdown.Markdown(
        extensions=["extra", "toc", "tables", "fenced_code", "codehilite", "pymdownx.arithmatex"],
        extension_configs={
            "toc": {"permalink": True, "slugify": slugify},
            "pymdownx.arithmatex": {"generic": True},
        },
        output_format="html5",
    )
    rendered = md.convert(path.read_text(encoding="utf-8"))
    return rendered.replace('href="http', 'target="_blank" rel="noopener noreferrer" href="http')


def nav(current: str | None) -> str:
    links = [
        ("Home", "index.html"),
        ("Usage", "usage.html"),
        ("Theory", "theory.html"),
        ("Algorithms", "index.html#algorithms"),
        ("GitHub", REPO_URL),
    ]
    items = []
    for label, href in links:
        active = ' aria-current="page"' if label == current else ""
        target = ' target="_blank" rel="noopener noreferrer"' if href.startswith("http") else ""
        items.append(f'<a href="{href}"{target}{active}>{label}</a>')
    return "\n".join(items)


def page(title: str, body: str, current: str | None = None) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="PennyLane-based quantum machine learning workflows by Sid Richards.">
    <meta property="og:type" content="website">
    <meta property="og:title" content="{html.escape(title)} | Quantum Machine Learning">
    <meta property="og:description" content="Research-grade PennyLane QML package with classifiers, regressors, kernels, QCNNs, autoencoders, and benchmarks.">
    <meta property="og:url" content="{SITE_URL}">
    <meta name="theme-color" content="#0f5364">
    <title>{html.escape(title)} | Quantum Machine Learning</title>
    <link rel="stylesheet" href="styles.css">
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  </head>
  <body>
    <a class="skip-link" href="#top">Skip to main content</a>
    <header class="site-header">
      <a class="brand" href="index.html" aria-label="Quantum Machine Learning home">
        <span class="brand-mark">SR</span>
        <span>Quantum Machine Learning</span>
      </a>
      <nav class="nav-links" aria-label="Primary navigation">{nav(current)}</nav>
    </header>
    <main id="top" tabindex="-1">{body}</main>
    <footer class="site-footer">
      <span>&copy; 2026 Sid Richards</span>
      <a href="#top">Back to top</a>
    </footer>
  </body>
</html>
"""


def tag_html(tags: list[str]) -> str:
    return "".join(f"<span>{html.escape(tag)}</span>" for tag in tags)


def card(title: str, description: str, href: str, tags: list[str] | None = None) -> str:
    tags_block = f'<div class="tags">{tag_html(tags)}</div>' if tags else ""
    return f"""
      <article class="project-card">
        <div>
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(description)}</p>
        </div>
        {tags_block}
        <div class="card-links"><a href="{html.escape(href)}">Read</a></div>
      </article>
    """


def home() -> str:
    algorithm_cards = "\n".join(card(*item) for item in ALGORITHMS)
    doc_cards = "\n".join(
        card(label, description, output) for label, _, output, description in DOCS[1:4]
    )
    body = f"""
      <section class="hero section">
        <div class="hero-copy">
          <p class="eyebrow">PennyLane quantum machine learning</p>
          <h1>Quantum Machine Learning</h1>
          <p class="hero-text">
            Modular research-grade workflows for variational classifiers, regressors,
            quantum kernels, QCNNs, autoencoders, metric learning, and deterministic
            benchmark comparisons.
          </p>
          <div class="hero-actions" aria-label="Project links">
            <a class="button primary" href="{REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a class="button" href="{PYPI_URL}" target="_blank" rel="noopener noreferrer">PyPI</a>
            <a class="button" href="usage.html">Usage</a>
            <a class="button" href="theory.html">Theory</a>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-visual" aria-hidden="true">
            <svg viewBox="0 0 520 360" role="presentation" focusable="false">
              <defs><pattern id="hero-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" /></pattern></defs>
              <rect class="visual-bg" width="520" height="360" rx="8" />
              <rect class="visual-grid" width="520" height="360" rx="8" fill="url(#hero-grid)" />
              <g class="visual-circuit">
                <path d="M72 86h344M72 146h344M72 206h344M72 266h344" />
                <path d="M168 86v120M276 146v120M360 86v180" />
                <circle cx="168" cy="86" r="13" /><circle cx="168" cy="206" r="13" />
                <circle cx="276" cy="146" r="13" /><circle cx="276" cy="266" r="13" />
                <circle cx="360" cy="86" r="13" /><circle cx="360" cy="266" r="13" />
                <path d="M121 68v36M103 86h36M103 128l36 36M139 128l-36 36" />
                <rect x="218" y="66" width="52" height="40" rx="8" />
                <rect x="400" y="126" width="52" height="40" rx="8" />
              </g>
              <g class="visual-labels">
                <text x="64" y="322">VQC</text><text x="150" y="322">VQR</text>
                <text x="230" y="322">QML</text><text x="314" y="322">QCNN</text>
                <text x="410" y="322">PyPI</text>
              </g>
            </svg>
          </div>
          <aside class="focus-panel" aria-label="Package focus areas">
            <h2>Focus Areas</h2>
            <ul>
              <li>Package-first PennyLane workflows</li>
              <li>Deterministic experiments and saved artifacts</li>
              <li>Classical baselines for direct comparison</li>
              <li>Notebook clients backed by reusable APIs</li>
            </ul>
          </aside>
        </div>
      </section>
      <section id="algorithms" class="section">
        <div class="section-heading">
          <p class="eyebrow">Implemented workflows</p>
          <h2>Algorithms</h2>
          <p>Each workflow has a public Python API, command-line entry points where appropriate, and focused documentation generated from this repository.</p>
        </div>
        <div class="project-grid">{algorithm_cards}</div>
      </section>
      <section id="docs" class="section">
        <div class="section-heading">
          <p class="eyebrow">Reference material</p>
          <h2>Documentation</h2>
          <p>Use the usage guide for API calls and CLI commands, the theory notes for mathematical background, and the changelog for release history.</p>
        </div>
        <div class="project-grid">{doc_cards}</div>
      </section>
      <section class="section contact-section">
        <div>
          <p class="eyebrow">Install</p>
          <h2>Use the package from PyPI or source</h2>
          <p>The package is published as <code>qml-pennylane</code> and can also be installed from GitHub for development.</p>
        </div>
        <div class="contact-actions">
          <a class="button primary" href="{PYPI_URL}" target="_blank" rel="noopener noreferrer">PyPI</a>
          <a class="button" href="{REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </section>
    """
    return page("Home", body, current="Home")


def documentation_page(label: str, source: Path) -> str:
    doc_nav = "".join(f'<a href="{output}">{html.escape(name)}</a>' for name, _, output, _ in DOCS)
    current = label if label in {"Usage", "Theory"} else None
    body = f"""
      <section class="section doc-layout">
        <aside class="doc-sidebar" aria-label="Documentation navigation">
          <p class="eyebrow">Documentation</p>
          <nav>{doc_nav}</nav>
        </aside>
        <article class="doc-content">{render_markdown(source)}</article>
      </section>
    """
    return page(label, body, current=current)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copyfile(ROOT / "docs/pages/styles.css", OUT / "styles.css")
    (OUT / "index.html").write_text(home(), encoding="utf-8")
    for label, source, output, _ in DOCS:
        (OUT / output).write_text(documentation_page(label, source), encoding="utf-8")


if __name__ == "__main__":
    main()
