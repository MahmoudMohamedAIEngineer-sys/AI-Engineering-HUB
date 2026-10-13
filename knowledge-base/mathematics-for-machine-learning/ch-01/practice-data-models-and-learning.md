# Practice: Data, Models, and Learning

Source topics for all problems: `ml.foundations.data-models-learning`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [ml.foundations.data-models-learning]
Name the three core components of a machine learning system and give one example failure caused by each component.

**Solution.** Data (the measurements), model (the parametrized function or distribution), learning (the procedure choosing parameters by optimizing a loss). Failures: stale 2015 messages = data; one-keyword rule family = model; training stopped after one huge-step pass = learning. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

### Problem 2 (recall)
source_topics: [ml.foundations.data-models-learning]
The phrase 'machine learning algorithm' is ambiguous in the book. State its two meanings and say which one gradient descent belongs to.

**Solution.** Predictor (the trained function making predictions) vs training procedure (the system adapting parameters). Gradient descent is a training procedure, i.e. part of learning, not the model. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

### Problem 3 (computation)
source_topics: [ml.foundations.data-models-learning]
For data (x,y) = (10,21),(15,29),(20,42) and model f(x)=t0+t1*x, compute the squared-error sum for (t0,t1)=(0,2) and (1,2). Which is better under this loss?

**Solution.** (0,2): errors (1,-1,2), SSE=1+1+4=6. (1,2): errors (0,-2,1), SSE=0+4+1=5. (1,2) is better under squared-error loss. Reproduce with the topic's code. [DERIVED]

### Problem 4 (computation)
source_topics: [ml.foundations.data-models-learning]
A lookup-table memorizer stores all 3 training pairs above. What is its training SSE, and why does the book's definition still not call this successful learning?

**Solution.** Training SSE = 0 (it returns stored labels). Success is defined by performance on unseen data; a lookup table has no rule for a new x (e.g. x=12), so generalization is undefined/poor. Memorizing is not the goal. [BOOK: mathematics-for-machine-learning, ch.1, §1.1]

### Problem 5 (derivation)
source_topics: [ml.foundations.data-models-learning]
Argue from the definitions (no formulas needed) why adding more data cannot fix a model class that is too restrictive, using lines fitted to points lying exactly on a parabola.

**Solution.** Every model in the class is a line; a parabola is not a line, so for data on a parabola at >=3 distinct x positions no line passes through all points. More data adds more points a single line must miss. The limitation is in the model component, independent of data quantity and of the learning procedure. [DERIVED from §1 definitions]

### Problem 6 (application (ML))
source_topics: [ml.foundations.data-models-learning]
A churn model performs badly. Logs show features were joined on a wrong key so 40% of rows have another customer's features. Which component failed, and what is the fix class?

**Solution.** Data component: the numerical conversion/assembly of measurements is wrong. Fix = correct the join and add data validation; changing model or optimizer cannot repair wrong measurements. [DERIVED]

### Problem 7 (application (ML))
source_topics: [ml.foundations.data-models-learning]
A team doubles training epochs but validation accuracy never improves; the model is a single linear layer on raw pixels for a task needing spatial structure. Which component is the bottleneck and why won't more learning help?

**Solution.** Model component: the function family cannot express the needed spatial patterns. Learning only chooses parameters inside the family; more epochs search the same inadequate family harder. Fix = change model family (e.g. convolutions), covered later in the DL topics. [DERIVED]

### Problem 8 (interview)
source_topics: [ml.foundations.data-models-learning]
Interview-style: 'Explain machine learning in one minute.' Give a model answer and a weak-answer example.

**Solution.** Model answer: 'We represent observations as vectors (data), choose a parametrized function or distribution (model), and choose its parameters by optimizing a loss on training examples (learning); success means low loss on unseen data, not the training set.' Weak answer: 'The computer learns patterns from data by itself' — no parameters, no loss, no generalization criterion. source_topics: [ml.foundations.data-models-learning]
