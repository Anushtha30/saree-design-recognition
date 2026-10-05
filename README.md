# Saree Design Recognition & Retrieval

A deep learning-based computer vision project for **recognizing and retrieving visually similar saree designs** while reducing the influence of saree colour and focusing more on the underlying design, motifs, and patterns.

The system allows a user to provide a query saree and retrieves similar designs from a gallery using **feature embeddings and cosine similarity**. A similarity threshold can be adjusted to control how closely the retrieved designs should match the query.

### Key Features

* Saree design and motif recognition
* Colour-invariant design retrieval
* Feature-based image similarity
* Cosine similarity ranking
* Configurable similarity threshold
* Interactive saree retrieval interface
* Gallery-based visual search

### How It Works

1. A saree image is provided as the query.
2. Visual features representing its design and motifs are extracted.
3. The query representation is compared with saree representations in the gallery.
4. Cosine similarity is calculated between the feature representations.
5. The gallery is ranked according to similarity.
6. Designs above the selected similarity threshold are considered relevant matches.

### Tech Stack

* Python
* Computer Vision
* Deep Learning
* Image Feature Extraction
* Cosine Similarity
* HTML/CSS/JavaScript
* Netlify

### Use Cases

This project can be used for **fashion image search, saree catalog management, e-commerce recommendation systems, traditional textile classification, and visual similarity search**.

> Note: The hosted interface is an interactive illustration of the intended retrieval behaviour. Actual model evaluation and numerical results are generated through the project's evaluation pipeline.
