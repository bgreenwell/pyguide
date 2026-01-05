# Product Guidelines: pyguide

## Documentation Standards
- **Technical Precision:** Documentation and code comments must prioritize mathematical and algorithmic accuracy. When implementing GUIDE-specific logic (like the Chi-square variable selection or Wilson-Hilferty approximation), refer to the source academic papers (e.g., Loh 2002, 2009).
- **Clear Terminology:** Use consistent terminology aligned with both the GUIDE literature and the scikit-learn ecosystem.
- **Reference-heavy:** Where appropriate, cite specific sections of the included resource PDFs (`classification.pdf`, `regression.pdf`, etc.) in the docstrings.

## Code Quality
- **Scikit-Learn Compliance:** Strictly follow scikit-learn's API conventions (`fit`, `predict`, `transform`, `get_params`, `set_params`).
- **Performance:** While correctness is paramount, aim for efficient implementations of statistical tests using `numpy` and `scipy`.
- **Test-Driven:** Prioritize unit tests that verify the statistical properties of the tree splits, not just the output shape.
