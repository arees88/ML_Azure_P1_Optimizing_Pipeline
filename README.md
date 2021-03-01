# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

The main steps of the project are in the diagram below:

![image](https://user-images.githubusercontent.com/60096624/109424630-41e19180-79dc-11eb-91c8-fed87d5b1d7e.png)

## Summary

### Problem Statement

The dataset provided for the project is related to the direct telemarketing campaigns of the Portuguese banking institution.
The marketing campaigns were based on phone calls.
The clients were contacted in order to asses if they would subscribe or not to the product - the bank long-term deposits.
The client's answer was recorded as 'yes' or 'no' respectively in the data records (column y). 

The objective of the project is to use the data to predict if a person will subscribe to the long-term deposit with the bank or not.

The dataset used for the project contains almost 33 thousand of records with 20 features.
The snapshot of the dataset is below: 

![image](https://user-images.githubusercontent.com/60096624/109403410-5a5a9900-7955-11eb-9812-c3806b9a8ffe.png)

### Solution

To solve the problem we need to build a classifier to predict if the customer of the bank will subscribe to a long-term deposit with the bank or not.

First we built and optimized an Azure ML pipeline using the Python SDK and a Scikit-learn **LogisticRegression** model.
The model was then optimised using **Azure HyperDrive** option to find the best hyperparameters for the logistic regression.

As the second option we used the **Azure AutoML** to automaticaly explore different classifaction models to identify the best one and to compare it to the LogisticRegression model.

After comparing the performance of the classifiers the best performing model was **VotingEnsemble** which was identified by Auto-ML run and performed slightly better than the LogisticRegression optimised with HyperDrive.

## Scikit-learn Pipeline

To construct the ML pipeline we used the Scikit-learn Logistic Regression classification model.
A custom script `train.py` was used to define the model specific functions.

The script required additional code to import the data from the specified URL (CSV file).
The data was imported using `TabularDatasetFactory` to create the `TabularDataset` object.
The custom `clean_data()` function was applied to preprocess the data before `train_test_split()` was called to split the data into train and test sets.

Two input arguments were configured in the `main()` function: ***C*** (inverse of regularization factor) and ***max_iter*** (maximum number of iteration).
These two model parameters were then used to define the hyperparameter sampler for the HyperDrive run.

The `train.py` script was used to create the estimator as follows:
```
est = SKLearn(  source_directory = "./",
                entry_script="train.py",
                compute_target=compute_cluster,
                vm_size='STANDARD_D2_V2'
)
```
The `SKLearn` estimator, together with the hyperparameter sampler and early stopping policy, were then used to define the HyperDrive run:
```
hyperdrive_config = HyperDriveConfig(   estimator = est,
                                        hyperparameter_sampling = ps,
                                        policy = policy,
                                        primary_metric_name = 'Accuracy',
                                        primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                                        max_total_runs = 20,
                                        max_concurrent_runs = 4)
```
The primary metric for hyperparamters tuning was defined as "Accuracy" with the goal of maximizing it.

The below screenshots show the completed HyperDrive run, including the list of sampler combinations (Tags) that were executed:

![image](https://user-images.githubusercontent.com/60096624/109431391-44ed7980-79fe-11eb-9d27-d70e41ad8d27.png)

![image](https://user-images.githubusercontent.com/60096624/109431410-5f275780-79fe-11eb-9d80-21606e32de1d.png)

### Hyperparameter Sampler

Azure ML supports three types of parameter sampling - Random, Grid and Bayesian sampling.
I have chosen Random Parameter Sampling because it is faster and supports early termination of low-performance runs.
It supports discrete and continous hyperparameters. 

In the train.py script we defined two parameters we can pass to create the LogisticRegression model:
```
parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
```
As the ***C*** parameter is of type float, I have used `uniform` distribution over a continuous range of values between 0.5 and 1.0.

The ***max_iter*** parameter is of type integer and I have used `choice` to specify four discrete values in the sampler as follows: 
```
ps = RandomParameterSampling({
        "--C":         uniform(0.5, 1.0),
        "--max_iter" : choice(50, 100, 150, 200)
})
```
In random sampling, hyperparameter values are chosen randomly, thus saving a lot of computational efforts.
It can also be used as a starting sampling method as we can use it to do an initial search and then continue with other sampling methods.

### Eearly Termnination Policy

The purpose of early termination policy is to automatically terminate poorly performing runs so we do not waste time and resources for the experiment. 

There are a number of early termination policies such as BanditPolicy, MedianStoppingPolicy and TruncationSelectionPolicy. 
For the project I have chosen the **BanditPolicy** based on the slack factor which is the mandatory parameter.
I have also specified optional parameters, evaluation interval and delay evaluation as follows:
```
policy = BanditPolicy (slack_factor = 0.1, evaluation_interval = 1, delay_evaluation = 5)
```
The above policy basically states to check the job at every iteration after the initial delay of 5 evaluations. 
If the primary metric (accuracy) falls outside of the top 10% range, Azure ML will terminate the job. 

The early termination policy ensures that only the best performing runs will execute to completion and hence makes the process more efficient.

## AutoML

Once HyperDrive run is completed, we want to compare it to Azure AutoML run.
In the secend part of the project we import the data from the specified URL again, clean it, but this time we pass the cleaned data to AutoMLConfig.

The following configuration was used for the AutoML run:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task = 'classification',
    primary_metric = 'accuracy',
    training_data = ds,
    label_column_name = 'y',
    n_cross_validations=6,
    enable_onnx_compatible_models=True
)
```
The AutoML experiment ended after 30 minutes as it reached the stopping criteria (`experiment_timeout_minutes=30`).

During this time AutoML performed 28 iterations evaluating a set of diffrent models. The best performing model was `VotingEnsemble` with accuracy `0.9176`:
```
ITERATION   PIPELINE                                       DURATION      METRIC      BEST
       27   VotingEnsemble                                 0:01:06       0.9176    0.9176
```
Here is the screenshot of the completed AutoML experiment with the best model summary:

![image](https://user-images.githubusercontent.com/60096624/109431457-ac0b2e00-79fe-11eb-8e4c-8a4ecaf09d63.png)

## Pipeline comparison

The Scikit-learn Logistic Regression model was tuned using HyperDrive.
After 20 iterations HyperDrive run identified the following hyperparameter values as those that gave the highest accuracy for the Scikit-learn model of **0.90728**:
```
['--C', '0.5171872582703092', '--max_iter', '200']
```
The best model identified using the AutoML option was the **VotingEnsemble** classifier which gave slightly better accuracy of **0.91763**.

#### Reason of difference

As part of the project we explored and compared the HyperDrive option, of finding the best hyperparameters for one logistic regression model from the Sklearn library,
with the AutoML option which explored many different types of classifaction models to find the best performing one. 

As such we are comparing a clasic logistic regression model from the Sklearn library, with a diffrent type of model architecture which is voting ensemble.
It is reasonable to expect that diffrent models give diffrent performance metrics.
In addition in both cases we restricted the number of iterations when searching for best metrics, so potentially we have not achieved the optimal performance from both options.

## Future work

As mentioned in the previous section, both experiments in the project had restricted number of iterations when searching for best model performance. 

In the case of HyperDrive experiment we used Random sampling and restricted the number of iterations to 20. 
Using higher number of iterations with more Random sampler choices may help with finding a set of hyperparameters that give better performance.
To make sure that we don't miss the best performing hyperparameter settings we could swith to Grid sampling instead.
Choosing a diffrent early termination policy may also help by providing savings without terminating promising jobs.
For example using the more conservative Median Stopping Policy rather than BanditPolicy.  

With the AutoML option we restricted the experiment time to 30 minutes which allowed for 28 models to be explored. 
Increasing the experiment time would potentially allow to find another, better performing model. 

We could also explore selecting diffrent dataset features for training.
In the original dataset a large set of 150 features related to the clients, the product and the social-economic attributes were collected. 
A semi-automatic feature selection was applied to the original data in the modeling phase that allowed to select a reduced set of features.
The dataset used for the project contains a reduced set of 20 features. It might be worth exploring if a diffrent selection of features would give better results.

## Proof of cluster clean up
Once the project is completed the compute cluster needs to be dealocated.
Here is the image of cluster marked for deletion performed after the notbook was saved:

![image](https://user-images.githubusercontent.com/60096624/109425393-f16c3300-79df-11eb-8f8a-8e7e76fce0a5.png)

## References

- S. Moro, P. Cortez and P. Rita, February 2014, "A Data-Driven Approach to Predict the Success of Bank Telemarketinghttp" [PDF] (http://repositorium.sdum.uminho.pt/bitstream/1822/30994/1/dss-v3.pdf)
- Kaggle competition, H. Yamahata, Bank Marketing, UCI Dataset, (https://www.kaggle.com/henriqueyamahata/bank-marketing)
- Hyperparameter tuning with Azure Machine Learning (https://docs.microsoft.com/azure/machine-learning/service/how-to-tune-hyperparameters#specify-an-early-termination-policy)
- Scikit-learn Logistic Regression (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
