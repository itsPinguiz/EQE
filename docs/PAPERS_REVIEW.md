### 1. Summary Table of XAI Methods

| Paper / Reference                              | XAI Method(s) Proposed/Used                                     | Type of Explanation                                                | Evaluation Approach                                                                                             |
| :--------------------------------------------- | :-------------------------------------------------------------- | :----------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **0747.pdf** (XAI Toolkits Survey)             | Saliency, GradCAM, LIME, SHAP                                   | Feature Attribution, Saliency Maps                                 | Functionally-grounded (analyzes Co-12 criteria coverage like correctness, completeness across 17 toolkits).     |
| **1910.02065v3.pdf** (Verification Framework)  | LIME, SHAP, L2X                                                 | Feature Attribution (Additive & Selective)                         | Functionally-grounded (synthetic evaluation of zero-contribution features).                                     |
| **2023.eacl-demo.29.pdf** (`ferret`)           | LIME, SHAP, Gradient, Integrated Gradients                      | Feature Attribution                                                | Functionally-grounded (Comprehensiveness, Sufficiency, LOO) & Human-grounded (Plausibility via IOU, Token F1).  |
| **2106.12543v4.pdf** (`XAI-BENCH`)             | SHAP, LIME, MAPLE, L2X, SHAPR, BreakDown                        | Feature Attribution                                                | Functionally-grounded (Faithfulness, Monotonicity, ROAR, Infidelity on synthetic ground-truth data).            |
| **2206.11104v5.pdf** (`OpenXAI`)               | LIME, SHAP, Vanilla Gradients, SmoothGrad, Integrated Gradients | Feature Attribution                                                | Functionally-grounded (Ground-truth & Predictive Faithfulness, Stability, Fairness).                            |
| **NeurIPS_2019...** (Infidelity & Sensitivity) | Grad, IG, GBP, SHAP, SmoothGrad                                 | Feature Attribution                                                | Functionally-grounded (Infidelity, Sensitivity) & Human-grounded (User study to infer ground truth).            |
| **NeurIPS_2021...** (BayesLIME/SHAP)           | BayesLIME, BayesSHAP                                            | Feature Attribution with Uncertainty                               | Functionally-grounded (Credible intervals, Lipschitz stability) & Human-grounded (Digit guessing task).         |
| **paper6.pdf** (Faithfulness Framework)        | Decision Trees, Anchors, LIME, SHAP, Counterfactuals            | Scoped Rules, Feature Attribution, Counterfactuals                 | Functionally-grounded (Consistency, Sufficiency metrics).                                                       |
| **paper7.pdf** (Robust Explanations)           | Gradient, Grad x Input, IntGrad, GBP, LRP                       | Saliency Maps                                                      | Functionally-grounded (Robustness to input/model manipulation via Pearson Correlation/SSIM).                    |
| **paper9.pdf** (Adversarial Robustness)        | Grad, IG, EG, SHAP, LOO, CFX, Greedy-AS                         | Feature Attribution / Subset                                       | Functionally-grounded (Adversarial Robustness, Insertion/Deletion) & Human-grounded (Keyword user study).       |
| **peerj-cs-479.pdf** (`LEAF`)                  | LIME, SHAP                                                      | Local Linear Explanations                                          | Functionally-grounded (Conciseness, Local Concordance, Local Fidelity, Reiteration Similarity, Prescriptivity). |
| **s10618-023...** (XAI Survey)                 | LIME, SHAP, LRP, Grad-CAM, TCAV, Prototypes, Counterfactuals    | Feature Importance, Saliency, Concept, Prototypes, Counterfactuals | Systematic Review covering Functionally, Application, and Human-grounded evaluations.                           |

---

### 2. Deep Dive: Current Evaluation Metrics

**Quality & Reliability (Functionally-Grounded)**
To prove mathematically and logically that explanations make sense, researchers employ computational proxy measures without human intervention. The core dimensions evaluated are:
*   **Fidelity / Faithfulness**: This measures how well the explanation mimics the true black-box model's behavior. Methods include:
    *   *Comprehensiveness & Sufficiency (AOPC)*: Tracking the change in prediction probability when important features (identified by the explainer) are either removed or kept as the sole inputs. 
    *   *Insertion & Deletion / ROAR*: Progressively adding or removing top features and measuring the area under the probability curve. 
    *   *Prediction Gap (PGI/PGU)*: Measuring the prediction gap when perturbing strictly the important (or unimportant) features.
*   **Robustness & Stability**: These metrics test if explanations remain consistent when inputs are slightly perturbed but the model's prediction remains unchanged. 
    *   *Local Lipschitz Estimation*: Calculates the maximum change in the explanation relative to the change in input.
    *   *Relative Output/Representation Stability (ROS/RRS)*: Measures explanation variance relative to changes in the model's output or internal representations.
    *   *Infidelity & Sensitivity*: Measures how much the explanation captures the predictor's response to significant, targeted perturbations (infidelity), versus infinitesimal random perturbations (sensitivity).
    *   *Adversarial Robustness*: Verifies if adversarial attacks are highly effective when applied to features the explanation deemed *relevant*, and ineffective on *irrelevant* features.
*   **Consistency & Sanity Checks**: 
    *   *Reiteration Similarity*: Ensures that applying a stochastic explainer (like LIME or SHAP) multiple times on the same input yields the same feature set.
    *   *Sanity Checks / Model Randomization*: Destroys the learned weights of the neural network layer by layer; a reliable explainer should drastically change its output if the model is randomized.

**Usefulness & Human-Centricity (Human/Application-Grounded)**
To assess if explanations actually benefit end-users, studies use specific cognitive tasks and plausibility checks:
*   **Plausibility**: Automated comparisons against human-annotated rationales (e.g., using Intersection-Over-Union or Token-level F1 scores) to see if the model's highlighted features match what a human would highlight.
*   **Task-Based User Studies**: 
    *   *Feature Masking*: Important features identified by the explainer are hidden from users. If the explainer accurately identified the most critical features, the human's ability to guess the correct output (e.g., classifying a MNIST digit) should drop significantly.
    *   *Ground-Truth Inference*: Users are shown explanations and asked to identify which specific component of the input (e.g., a text caption vs. an image) the model relied on to make its prediction. 
    *   *Prescriptivity (Actionability)*: Used in the `LEAF` framework, this evaluates if an explanation can serve as a successful "recipe" for a human to change an instance's classification (e.g., identifying exactly how much to lower blood pressure to change a risk prediction).
