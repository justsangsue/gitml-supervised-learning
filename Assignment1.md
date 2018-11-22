# **CS 7641 Machine Learning**

## Decision Trees
> **Gini impurity**: Used by the **CART** (classification and regression tree) algorithm for classification trees, Gini impurity is a measure of ***how often*** a randomly chosen element from the set would be ***incorrectly labeled*** if it was randomly labeled according to the *distribution of labels in the subset*.

Gini impurity is pretty much the same as Information Gain Entropy (defined below), according to [this answer](https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy). **Then why we need those in decision trees?**
> **Information Gain**: Used by the ID3, C4.5 and C5.0 tree-generation algorithms. Information gain is based on the concept of entropy from information theory.

Information gain is used to decide **which feature to split on** at each step in building the tree. *So why we want to "split"? What does "split" mean?*

In dicision tree learning algorithms, what we want is to build a "Tree" that can give predictions based on input features. An example is given below:

<center>
![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png) 
</center>


From the example, we can easily understand **split**: In a dataset, each entry (observation, data point...) has several attributes (for the definition of "attributes" or "features", I like [this answer](https://stackoverflow.com/questions/19803707/difference-between-dimension-attribute-and-feature-in-machine-learning)) and a result (aka. y, output, value...should be a boolean value 0/1). After asking each question (check a attribute), we can split the whole dataset into two groups, "group-yes" and "group-no". Generally, the distribution of 0/1 in each group is uneven (if not, the attribute is useless). Then we can ask another question (or check another attribute) in each group, so that we can further split data. Ideally (not common I think), data could be splitted perfectly into groups with all-yes and all-no. At this time, we finish building this tree. (This process is like drawing lines to split colored dots on a paper)

<center>
![The 3rd picture](https://upload.wikimedia.org/wikipedia/commons/2/25/Cart_tree_kyphosis.png)
</center>

After knowing this, now we can go back to the **Gini impurity** and **Information Gain Entropy**. 
___
### Now it's time to perform decision trees on our own dataset!!!
I chose a medical dataset from [kaggle](![kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting)). In this dataset, the column needs to be predicted is "CLASS", whose possible values are 0 and 1.

The ML code is stolen from [here](https://github.com/vicflair/cs7641/tree/master/hw1/code). 

The first thing I notice is that the data contains **many non-numeric columns**.

<del>
One idea to solve is to make a set to **find all unique data and then enumerate each value**. e.g.: if we have a variable 'city', and the following dataset [(n0, Tokio), (n1, Rome),  (n2, Rome), (n3, London)], we can transform it as follows: [(n0, 1,0,0), (n1,0,1,0), (n2,0,1,0), (n3,0,0,1)]

<del>
That is a good idea. Since I am using sklearn in python to perform the machine learning, so I need to know whether sklearn can deal with **multidimentional data**. (I decide to just leave it aside and consider it if any problem occurs.)

After looking for more tutorials, I decided to deal with non-numeric catagorical data by adding new columns and filled with 0/1. (Now I know this is called *"One Hot Coding"*)

The first thing I am going to do is **pre-processing the data**:

1. Remove unrelated features
2. Convert non-numeric to numeric (by adding new columns)

Well, after three days work, the data is finally cleaned and I jave finished decision trees, boosted decision trees (AdaBoost) and doing kNN now...
Believe it or not, I have finished the project and submit it! But with so many things not totally understood.