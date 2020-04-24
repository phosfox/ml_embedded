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
    - like kmeans, mse
---
# Learning methods

## Supervised learning

> Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. Y = f(x).

The goal of this method is to train the mapping function good enough that it can predict the output variable (Y) for new input data (X). 
It is called **supervised** learning because the process of the algorithm learning from a training dataset can be seen as a teacher supervising the learning process.
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

[Supervised learning](https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)

## Unsupervised learning

## Reinforcement