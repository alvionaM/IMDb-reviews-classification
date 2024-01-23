<div id="header" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://media.giphy.com/media/kksFTNC9TL93AnhuLi/giphy.gif">
    <img alt="Hello World!" src="https://media.giphy.com/media/89jRrowcuHEG0OFavV/giphy.gif" height="200"">
  </picture>
</div>

## Description📌
<p align="justify">This repository hosts a project of sentiment analysis performed on IMDb movie reviews using Logistic regression, Bernoulli Naive Bayes classifier and a biGRU RNN. The purpose of this project is to showcase and examine performance differences between the aforementioned methods of classification, using metrics such as accuracy score, precision score, recall score and f1 score. </p>

<p align="justify">The datased used is the <a href="https://keras.io/api/datasets/imdb/">IMDb movie review sentiment classification dataset</a>. It consists of 25,000 movies reviews from IMDb, labeled by sentiment (positive/negative). <br>
After fetching and transforming the data, we implement the Logistic regression algorithm and a custom Bernoulli Naive Bayes classifier. We also construct a bidirectional GRU cell RNN with 2 layers. <br>
Then, we proceed to making comparisons of our custom approaches with the corresponding Scikit-Learn implementations mainly by plotting learning curves and printing classification reports to observe their behavior for both training and testing data. <br>
Lastly, we compare the behavior and performance of the bidirectional GRU cell RNN with those of the other two algorithms.</p>

The classification methods featured are:

| custom                    | scikit-learn                                        |              
|---------------------------|-----------------------------------------------------|
| NaiveBayesClassifier      |  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html">BernoulliNB</a>                                        | 
| CustomLogisticRegression  |  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">LogisticRegression</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html">SGDClassifier</a>                  |
| BiGRU_RNN                 |                                                     |


<br>

## Interact with the project 💻
This project consists of a Jupyter Notebook with all code cells already ran, so that you can easily examine the learning curves, classification reports, comparison heatmaps and other results. Also, there is a Report included summarizing some implementation details and conclusions. It's written in Greek (as the Artificial Intelligence course was offered in Greek).
> If you would like to rerun the Jupyter Notebook code cells, **please note** that some training processes may require plenty of time to complete depending on your machine characteristics.

<br>

## Implementation Details📜
### ● Bernoulli Naive Bayes Classifier (BernoulliNB)
<p align="justify">BernouliNB is implemented as a class (class NaiveBayesClassifier()) where <b>fit</b> and <b>predict</b> are defined as methods.</p>


- **fit** (x_train_binary, y_train)
Fit method is used for the computation (and assignment to the relevant class fields) of all probabilities required during prediciton:
    1. The probabilities $\color{#2c73cc}P(C=1), P(C=0)$ of positive and negative classes respectively have to be computed.
    2. Then, the conditional probabilities $\color{#a42574} P(feature_i = 1 | C=0), P(feature_i = 1 | C=1)$ should be known during prediciton to compute their product (Assumption of Independence). Obviously, there is no need to store $P(feature_i = 0 | C=0) = 1 - P(feature_i = 1 | C=0), P(feature_i= 0 | C=1) = 1 - P(feature_i = 1 | C=1)$, therefore reducing memory requirements. <br>
💡Laplace smoothing is used in the above computation to avoid zeroing of the entire classification probability due to zeroing of a single product term during prediction. Thus, +1 is added to the numerators of the above probabilities, while +2 is added to the denominators (if there are two possible values for each feature). <br>
    
    $`\boxed{P(C=0 | example) = \color{#2c73cc}P(C=0) \color{black}*\color{#a42574} \prod_{i=1}^{m}P (feature_i | C=0)}`$<br>
    $`\boxed{P(C=1 | example) = \color{#2c73cc}P(C=1) \color{black}*\color{#a42574} \prod_{i=1}^{m}P (feature_i | C=1)}`$
    
<br>

- **predict** (x_test_binary)
Predict method is used to compute the classification probabilities for the testing data given.
    - Algorithm:
      - ➡️ For each  instance of the testing set to be classified:
        - ➡️ For each feature of that instance:
            - ➡️ Compute the following probabilities:

    $`P(C=0 | feature_i) = P(C=0)*\color{#2c73cc}P(feature_i | C=0) =\begin{cases}
P(feature_i = 1 | C=0), if\; feature_i = 1,\\
1 - P(feature_i = 1 | C=0), if\; feature_i=0
\end{cases}`$

    $`P(C=1 | feature_i) = P(C=1)*\color{#2c73cc}P(feature_i | C=1) =\begin{cases}
P(feature_i = 1 | C=1), if\; feature_i = 1,\\
1 - P(feature_i = 1 | C=1), if\; feature_i=0
\end{cases}`$
    <br>
    ...which are then multiplied according to the formula, giving $P(C=0 | example)$ and $P(C=1|example)$
    The instance is classified in the class associated with the greatest probability.

---

### ● Logistic Regression
<p align="justify">Logistic Regression is implemented as a class (class
CustomLogisticRegression()) where <b>fit</b> and <b>predict</b> are defined as methods along with an auxilliary sigmoid function (<b>pos_category_sigmoid</b>). Moreover, <b>find_best_regularizator</b> is implemented to estimate the optimal regularization factor.</p>

- **fit** (x_train_binary, y_train)
    1. Data is splitted into training data and validation data, with validation percentage being 20%.
    2. All attributes' weights are initialized to 0. 
    <br>

    For a maximum number of iterations/epochs (n_iters) we randomly reorder the instances in the beginning of each  iteration (so that the steps towards weight convergence be independent from iteration to iteration) and for each instance we update the weights (and the bias factor) according to the formula:<br>
    $`\boxed{\vec{w} = (1-2*λ*η)*\vec{w}+η*\sum_{i=1}^m[y^{(i)}-P(C_+|\vec{x}^{(i)})]*\vec{x}^{(i)}}`$

Therefore, since weights are updated based on one example at a time, $\vec{x}^{(i)}$, we have Stochastic Gradient Ascent.
To terminate the process without exceeding the maximum number of iterations, the accuracy score is computed on validation data, using Early Stopping. If the accuracy does not improve in 20 iterations at most, the fitting process stops and the optimal weights that had been found are kept.
        
<br>

- **predict** (x_test_binary)
  - Algorithm: 
    - ➡️For each  instance of the testing set to be classified:
        - ➡️Compute the product of the weight vector (as learned during fitting) and the feature vector of the current example ($\vec{w} * \vec{x}$). The classification is done according to the sign of this product:

            $`(\vec{w} * \vec{x})=\begin{cases}
            pos (+) \rightarrow C=1\\
            neg (-)\rightarrow C=0
            \end{cases}`$
            <div align="center">
                <img src="../media/diagram.png" alt="Diagram" width="250">
            </div>

#### λ : regularization factor
<p align="justify">To find the optimal regularization factor (find_best_regularizator function) an iterative process takes place, where for a range of values of λ (from 1e-15 to 0.99 + 1e-15) accuracy score evaluations are performed on validation data and the optimal λ is returned along with other relevant information. If the accuracy score does not improve within 5 consecutive iterations/trials, then the process is terminated.</p>

<p align="justify">The optimal λ was proved to be the smallest of the above range (1e-15), and even after a trial in which λ was set to 0 the results were even better (i.e. without using
regularization). This is reasonable, since, as will be seen in the diagrams, there is not a high variance (overfitting) problem, as the learning curves of the training data and testing data
have converged. Therefore, any effort for further improvement would rather move towards the reduction of λ.</p>


#### η : learning rate
We use the typical value 0.001.

#### Number of maximum iterations
The value is set to 100, as it was experimentally observed that by using early stopping the iterations/epochs needed were generally less than 100.

---

### ● RNN 
<p align="justify">The bidirectional GRU cell RNN is implemented as a class () where <b>fit</b> and <b>predict</b> are defined as methods.</p>

- **Fit:**
Each time fit is invoked, the Neural Network is reconstructed (create_bi_GRU_RNN()) and compiled. Then, the standard fit (of keras model) is called to train it. The purpose is to not retain the data of previous calls, i.e. to "clear" the memory of the Neural Network.

- **Predict:**
Predict invokes the corresponding standard predict (of keras model).Then the predictions are converted from probabilistic to binary. This is done to be able to reuse the learning_curves function for our graphs.
<div align="center">
    <img src="../media/model.png" alt="biGRU RNN model">
</div>


## Contributors
- Alviona Mancho [<a href="https://github.com/alvionaM">alvionaM</a>]
- Christos Patrinopoulos [<a href="https://github.com/techristosP">techristosP</a>]
