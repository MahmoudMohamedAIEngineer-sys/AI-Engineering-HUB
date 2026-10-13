---
id: ml.foundations.data-models-learning
book: mathematics-for-machine-learning
chapter: 1
section: 1.1
tag: must_know
prerequisites: []
leads_to: [math.linalg.systems, math.prob.spaces, math.opt.gradient-descent]
---
# Data, Models, and Learning: Finding Words for Intuitions

## 1. What it is (plain-language definition, then the formal definition)

In plain language, a machine learning system does three things: it looks at examples (data), it contains an adjustable guess about how the world turns inputs into outputs (a model), and it adjusts that guess so the guess scores better on the examples (learning). [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

Formally, for the purposes of this course: **data** are represented as vectors in R^n, one vector per observation, with each coordinate recording one measured attribute. A **model** is either a function that maps inputs to predictions, or a probability distribution over possible data, controlled by parameters theta. **Learning** is the procedure that chooses theta by optimizing a utility (equivalently, minimizing a loss) measured on the training data, with the actual goal being good performance on unseen data, not on the training data itself. [BOOK: mathematics-for-machine-learning, ch.1, §1.1] The authors also warn that the same word is used for two different objects: "machine learning algorithm" can mean the predictor (the trained function that makes predictions) or the training procedure (the system that adapts the parameters). Keeping those two meanings separate removes a large fraction of beginner confusion, because questions like "is gradient descent the model?" then answer themselves: gradient descent belongs to learning, the network or regression function belongs to the model, and the examples belong to data. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

## 2. Why it exists (the problem it solves; what breaks without it)

Without this three-part vocabulary, every later discussion collapses into a single blurry word, "AI". When a system fails in production, the fix depends entirely on which component failed: bad measurements are a data problem, an inflexible function family is a model problem, and a good model family with badly chosen parameters is a learning (optimization) problem. Each failure has a different remedy — collect or clean different measurements, change the function family, or change the optimizer, loss, or amount of training. [DERIVED: the diagnostic split follows directly from the definitions in §1; verified by applying it to the worked example in §5 below, where each of the three failures is constructed separately.]

The vocabulary also explains why memorizing training answers is not learning. The definition of success is performance on data that was not used to choose the parameters. A system that stores every training input and looks up its label has optimized training performance perfectly while having no rule at all for a new input. The data/model/learning split makes this visible: such a lookup table is a degenerate model whose parameters are the dataset itself, and it exposes the missing ingredient — a model class that can interpolate between observed points. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

## 3. Intuition (a physical or visual analogy)

Think of fitting a curve through points on graph paper. The plotted points are the data: fixed, given, not yours to change. The family you choose — "straight lines only" or "parabolas" — is the model: it restricts the shapes you are allowed to end up with before you see any fitting. Sliding and tilting one chosen shape until it passes as close as possible to the points is learning: the shape's adjustable knobs (intercept and slope for a line) are the parameters, and "close" is measured by a rule you pick, such as total vertical distance. The authors use the related picture of climbing a hill, where height is the performance measure and learning walks uphill; in most of this course we will walk downhill on a loss instead, which is the same walk with the sign flipped. [BOOK: mathematics-for-machine-learning, ch.1, §1.1] The analogy earns its keep later: underfitting is choosing a line when the points curve, overfitting is choosing a shape so flexible it threads every point including the plotting errors, and both are statements about the model component, not about the amount of training effort.

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

There are no theorems in this section; the mechanics are the definitions and how they compose. Step 1: convert each observation to a vector x in R^D, where D is the number of recorded attributes; a dataset of N observations is then a collection {x_1, ..., x_N}, often stacked as rows of a matrix (Chapter 2 makes this precise). This conversion is an assumption, not a fact of nature: the book states it works with data that has already been converted to numerical form, so every conclusion downstream inherits the quality of that conversion. [BOOK: mathematics-for-machine-learning, ch.1, §1.1] Step 2: choose a model. In the function view, the model is a parametrized mapping f_theta from inputs to outputs; in the probabilistic view, it is a parametrized distribution p_theta over data. Both views name the same knob set, theta. Step 3: define a scalar utility or loss that scores parameter values on the training set; for a loss L, learning selects theta-star = arg min over theta of L(theta), the parameter value attaining the smallest loss. Step 4: judge the result on unseen data. The gap between training loss and unseen-data loss is the quantity the whole field worries about, and Chapter 8 of the book returns to these three components with full mathematical machinery once the tools exist. [BOOK: mathematics-for-machine-learning, ch.1, §1.2]

One more mechanical point the section insists on: a vector itself has three legitimate meanings — an array of numbers (computing view), an arrow with direction and magnitude (physics view), and an object that can be added to its own kind and scaled by a number (mathematical view). All three describe the same object in R^n; Chapter 2 builds the mathematical view, and analytic geometry in Chapter 3 reconnects it to arrows and angles. Confusing the views causes real errors: an embedding vector's coordinates, for instance, have no individual physical meaning even though the arrow picture still helps reason about closeness. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

Worked example A (the three components, with numbers). Suppose we record three houses: living area in hundreds of square feet and price in hundreds of thousands: (10, 21), (15, 29), (20, 42). Data: vectors x = (10, 15, 20) with labels y = (21, 29, 42) — in Chapter 9 notation, N = 3 observations. Model: lines, f_theta(x) = theta_0 + theta_1 x, so theta has two knobs. Learning rule: choose theta minimizing the sum of squared errors. Trying theta = (0, 2): predictions (20, 30, 40), errors (1, -1, 2), squared-error sum = 1 + 1 + 4 = 6. Trying theta = (1, 2): predictions (21, 31, 41), errors (0, -2, 1), sum = 0 + 4 + 1 = 5. The second parameter setting is better under this loss, and learning is exactly the search that finds the best such setting systematically instead of by guessing (Chapter 9 derives it in closed form; Chapter 7 does it iteratively). [DERIVED: arithmetic computed directly; re-verified by the code in §6.]

Worked example B (which component failed?). A spam classifier labels every message containing "free" as spam and everything else as not spam, and it misses most real spam. If the rule family "single keyword tests" is all the model can express, this is a model limitation: no training procedure can make a one-keyword rule represent "keyword combinations matter". If instead the model is logistic regression over many word features but it was trained on messages from 2015, the stale measurements are a data limitation. If the model and data are fine but training stopped after one pass with a huge step size, that is a learning limitation. Same symptom (missed spam), three different components, three different fixes. [DERIVED from the §1 definitions.]

## 6. Code (runnable, with saved output)

Pure-standard-library Python (DEC-005) that reproduces Worked Example A:

```python
xs = [10, 15, 20]
ys = [21, 29, 42]

def sse(theta0, theta1):
    total = 0
    for x, y in zip(xs, ys):
        err = y - (theta0 + theta1 * x)
        total += err * err
    return total

print("theta=(0,2):", sse(0, 2))
print("theta=(1,2):", sse(1, 2))
```
```text
theta=(0,2): 6
theta=(1,2): 5
```

## 7. Where it shows up in AI/ML (concrete uses)

Every later module maps onto this split: linear regression (book Chapter 9) is the line-fitting example made rigorous; PCA (Chapter 10) is a model of data as living near a low-dimensional subspace; Gaussian mixture models (Chapter 11) are the probabilistic-view model; support vector machines (Chapter 12) are a model plus a constrained-optimization learning rule. In LLM engineering the same split reads: pretraining corpus (data), transformer (model), next-token training and fine-tuning (learning) — and production failures are debugged by asking which of the three changed. [BOOK: mathematics-for-machine-learning, ch.1, §1.2, for the four pillars; the LLM mapping is DERIVED by direct analogy.]

## 8. Common mistakes and misconceptions

(1) Treating "the algorithm" as one thing, so that "the model is gradient descent" — gradient descent is a learning procedure, not a model. (2) Judging learning by training score; the definition of success is unseen-data performance. (3) Believing more data fixes a model-class limitation: a line stays a line no matter how many points you fit it to. (4) Assuming data arrive as numbers by nature; the numerical conversion is a human choice that discards information and must be audited. (5) Assuming the three views of a vector are interchangeable in every argument — coordinate-wise claims need the array view, closeness claims need the geometry view.

## 9. Interview angle (how it is asked, what a strong answer includes)

Common prompt: "Explain machine learning in one minute." A strong answer names data (vectors of measurements), a model (parametrized function or distribution), learning (optimizing parameters against a loss on training data), and the success criterion (generalization to unseen data), then gives one concrete failure per component. Weak answers say "the computer learns patterns from data" with no parameters, no loss, and no generalization criterion.

## 10. Visualization spec (what should be drawn or interactive, with the parameters the user can change)

Scatter plot of the §5 houses with a draggable line: sliders for intercept theta_0 (-5..5) and slope theta_1 (0..4); the squared-error sum updates live, and vertical residual segments are drawn so the learner sees the loss as geometry. Parameters the user can change: theta_0, theta_1, and a toggle that adds a noisy fourth house to show training fit vs unseen-point error.

## 11. Sources and verification (provenance tags and proof file paths)

Sources: [BOOK: mathematics-for-machine-learning, ch.1, §1.1] and [BOOK: mathematics-for-machine-learning, ch.1, §1.2] throughout; [DERIVED] items show their reasoning in place. Code executed by scripts/code_check.py (see proofs/gate-2-ch1.txt once the chapter gate runs). No web sources were needed for this book-derived topic.
