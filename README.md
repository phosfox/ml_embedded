# ml_embedded

## Seminar

The goal for this course is to introduce a procedure of Machine Learning
    
- classification scheme on constrained devices 
- preferably trained on the constrained device (If its not possible then train it on the notebook)
---
## Exam

- Presentation of a theory part
- Show what the model can recognize
- Each team should introduce to 2 algorithms of machine learning
    - like k-means, mse
---
# Learning methods

## Supervised learning

> Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. Y = f(x).

The goal of this method is to train the mapping function good enough that it can predict the output variable (Y) for new input data (X). 
It is called [**supervised** learning](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/) because the process of the algorithm learning from a training dataset can be seen as a teacher supervising the learning process.
Because of the training dataset we know the correct answers (labeling), the algorithm makes predictions interatively on the training data and is corrected by the teacher if it was wrong.
The learning process stops when the algorithm achieves an acceptable level of performance.

### Problems

Problems can be grouped into **regression and classification** problems:

- **Classification**: Output variable is a category, such as "red" or "blue" or "disease" and "no disease"
- **Regression** : Output variable is a real value, such as "dollars" or "weight"

Some common types of problems built on top of classification and regresion include **recommendation** and **time series prediction**.

### Popular examples of machine learning algorithms

- Linear regression for regression problems
- Random forest for cassification and regression problems
- Support vector machines for classification problems

## Unsupervised learning

> It is called [unsupervised learning](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/) if you only have input data (X) and no corresponding output variables. 

The goal for this algorithm is to model the underlying structure or distribution in the data in order to learn more **about** the data. Unlike supervised learning, there are no correct answers and there is no teacher for teaching the algorithm. The algorithm are left to their own devises to **discover** and **present** the interesting structure in the data. 

### Problems

Can be grouped into **clustering and association** problems:

- **Clustering**: The problem is where you want to discover the inherent groupings in the data
- **Association**: An association rule learning problem is where you want to discover rules that describe large portions of the data

### Popular examples of machine learning algorithms

- k-means for **clustering** problems
- Apriori algorithm for **association** rule learning problems

## Semi-Supervised learning

> Problems where you have a large amount of input data (X) and only some of the data is labeled (Y) are called [semi-supervised](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/) learning problems.

Therefore this kind of problems sit in between **supervised and unsupervised** learning. As an example, a folder is containing a bunch of photos and only a few of them are labeled but the majority is unlabeled. That means that many real world machine learning problems fall into this area because it can be **expensive** or **time-consuming** to label data. 

It is semi-supervised learning because you can use **unsupervised** learning techniques to discover and learn the structure in the input variables and you can use **supervised** learning techniques to make the best guess predictions for unlabeled data. You then feed that data back into the supervised learning algorithm as training data to use the model for predictions on new unseen data.

## Reinforcement learning

> [Reinforcement learning](https://openai.com/blog/openai-gym-beta/) (RL) is the subfield of machine learning concerned with decision making and motor control. It studies how an agent can learn how to achieve goals in a complex, uncertain environment 

Reinforcement learning has two goals:

- **Reinforcement learning is very general, it includes all problems that involve making a sequence of decisions**: For example, controlling a robots' motors so it is able to run and jump. Reinforcement learning can be **applied** to supervised learning problems with **sequential or structured** outputs.
- **Reinforcement learning algorithms have started to achieve good results in many difficult environments**

Reinforcement learning has two downsides:

- **Needs better benchmarks**: The equivalent for datasets in reinforcement learning are environments. Existing open-source environments do not have enough variety, are often difficult to set up and to use.
- **Lack of standardization of environments used in publications**: Subtle differences in the problem definition can **drastically** alter a task's difficulty. This makes it difficult to reproduce published research and to compare results from different papers.