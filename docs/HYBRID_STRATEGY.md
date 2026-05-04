### Complexity-Calibrated Local Concordance

**The Hybrid Intersection:** Quantitative Faithfulness (Functionally-grounded) + Explanation Conciseness (Human-grounded / Cognitive limit).

**The Targeted Gap:** Explanation methods often optimize mathematical fidelity while ignoring the cognitive load placed on human working memory, frequently evaluating feature attribution without constraints on how many features are presented to the user. 

**Theoretical Definition:** This metric combines the **Local Concordance** of an explanation (how well the white-box explanation mimics the black-box model locally) with a strict **Conciseness** constraint. Instead of allowing an explainer to use all features to perfectly mimic the black box, a penalty factor is applied to the mathematical fidelity score if the complexity of the explanation exceeds a human's working memory threshold limit (e.g., $K$ features). **The explanation is only considered highly scored if it can faithfully mimic the local decision boundary using a human-readable, highly condensed subset of features**.

**Practical Experimental Design:**
*   **Dataset:** Tabular clinical data (e.g., predicting heart risk from patient traits).
*   **AI Task:** Binary classification of high vs. low risk. 
*   **Human Task:** None required directly for the metric calculation, but the cognitive threshold $K$ is set based on human capacity (e.g., $K=4$). 
*   **Score Calculation:** Extract the top $K$ features from the explanation. Compute the Hinge loss between the original black-box prediction probability and the local linear explanation model's prediction *restricted strictly to those $K$ features*. A score of 1 indicates perfect local concordance within the human complexity limit, while a score approaching 0 indicates total disagreement.

