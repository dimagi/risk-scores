# RISK SCORES GENERATION

This repository provides example tooling for developing a risk score model, including the use of appropriate metrics for evaluation and examples of how to tune a model appropriately. The example is meant to illustrate the process as the details of any project will require some customization in approach.

The primary aim of using a machine learning model in public health is to tell us where to focus human efforts or resources and maximise our impact. This contrasts with a traditional economist or epidemiologist approach, which strives to establish causality and determine the exact factors that are responsible for the outcome.

Machine learning-based resource allocation allows us to identify clients or Front Line Workers (FLWs) who are at the highest risk for a particular (e.g., malnourishment in children) outcome and direct appropriate attention towards them to attempt to improve the outcome. For example, a predictive model that identifies high-risk pregnancies would allow an organization to intervene in due time, such as by focusing additional resources on the pregnant woman to ensure a safe pregnancy.

When applying machine learning algorthims to public health datasets that were collected in the field using a mobile health tool such as CommCare, DHIS2 we need to keep in mind that it is likely that the dataset is very noisy and highly imbalaced. Therefore when evaluating the performance of such algorthims we should carefully choose an appropriate metric. 

For example, for the classification tasks, a simple metric, such as accuracy, is unsuitable for evaluating where the occurrence of one category is small. If only 7% of pregnancies have an adverse outcome, we can achieve 93% accuracy if we classify every pregnancy as having a positive outcome. The specific metric used is covered in more detail in each of the project-specific sections.

To avoid the previously mentioned issue with accuracy as a metric, data scientists often use the AUC (area under curve), which is the two-dimensional area underneath the receiver operating characteristic (ROC) curve, as an evaluation metric. The ROC curve plots true positive rate against false positive rate in order to visualize the performance of a model in different scenarios. Therefore, we use AUC score in the code in this repository. 

## Dataset Loading and Saving Models
The dataset used for this analysis is the form submission data from CommCare applications. Each row in the dataset corresponds to a case and consists of all the historical form submission data for the particular case. In *build_model.py*, the dataset is loaded from a parquet file (you can similarly load a csv or sql file as well). 

The final trained model can be saved as pickle file at a desired location by setting the PATH variables in the *settings.py* file. You can then load your saved model, generate risk scores on your own dataset (formatted as one row per case) and save the risk scores as a csv file in *build_model.py* file.

## Submit Data to CommCare
Following the example script from this [repository](https://github.com/dimagi/submission_api_example), you can submit the risk scores to update the case properties in CommCare using the *submit_data.py* file. 

(If you would like to update existing cases, you can set the server_modified_on attribute for existing cases.)

## Usage
1. (Recommended) Create a virtual enviroment and activate it
```bash
$ python3 -m venv venv
$ source venv/bin/activate
```
3. Use the *requirements.txt* file to download all of the required libraries to run the script. cd to to the directory where the *requirements.txt* is located
```bash
$ pip install -r requirements.txt
```
4. Fill the variables in the *settings.py* file with the appropiate values, based on the previous step.
5. You can train a binary Random Forest Classifier by running: 
```bash
$ python3 build_model.py
```
5. You can generate risk scores for a case and update the case properties on CommCare HQ via the [Submission API](https://confluence.dimagi.com/display/commcarepublic/Submission+API) by running :

```bash
$ export CCHQ_PROJECT_SPACE=my-project-space
$ export CCHQ_CASE_TYPE=person
$ export CCHQ_USERNAME=user@example.com
$ export CCHQ_PASSWORD=MijByG_se3EcKr.t
$ export CCHQ_USER_ID=c0ffeeeeeb574eb8b5d5036c9a61a483
$ export CCHQ_OWNER_ID=c0ffeeeee1e34b12bb5da0dc838e8406
```

```bash
$ python3 submit_data.py sample_data.csv
```

