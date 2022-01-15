# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
### ResNet18 Model
- This project uses the ResNet-18 pre-trained model. 

- The ResNet-18 is a convolutional neural network that is 18 layers deep. This pre-trained version of the network is trained on more than a million images from the ImageNet database The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. 
- As a result, the network has learned rich feature representations for a wide range of images. This is why it is preferred for image classification here.

### Training Jobs
![Training Jobs](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/Training_jobs.png)
### Log Metrics
![Log Metrics](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/Log_Metrics_Graph.PNG)

### Hyperparameters
 Give an overview of the types of parameters and their ranges used for the hyperparameter search
 The hyperparameters that have been tuned are as follows:
 
 - **lr - Learning Rate** - If the LR is too small, overfitting can occur. Large lr help to regularize the training but if it is too large, the training will diverge.
 - **batch-size** - The batch size is limited by your hardware’s memory, while the learning rate is not. Small batch sizes add regularization while large batch sizes add less.
 - **epochs** - this value is key to finding the model that represents the sample with less error. One epoch leads to underfitting. As the number of epochs increases, and the model goes from underfitting to optimal to overfitting.

- The `learning rate` is a `ContinuousParameter`, the `batch-size` is a `CategoricalParameter` and the `epochs` is a `IntegerParameter` 

**Insert screenshots of the hyperparameter tuning and the best hyperparameter estimator**
### Hyperparameter Tuning
![Hyperparameter Tuning](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/Hyperparameter_training.png)

### Best Hyperparameters
![Best Hyperparameters](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/best_hyperparameters.png)

## Debugging and Profiling

- **Debugging and Profiling** was done with the help of the `sagemaker.debugger` module.
- The following rules were applied:
``` 
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
```
- **rule_configs** - A helper module to configure the SageMaker Debugger built-in rules with the 
Rule classmethods and and the ProfilerRule classmethods.
- **Rule** - The SageMaker Debugger Rule class configures debugging rules to debug your training job. The debugging rules analyze tensor outputs from your training job and monitor conditions that are critical for the success of the training job.

### Results
No errors arose during the session.
### Debugger Output
![Debugger Output](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/debugger_output.PNG)

If we did face errors, we can take a look at the CloudWatch logs to understand where we wrong and also have a look at the profiler/debugger report. Then we can look at the documentation to fix it.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
-To deploy the model, it has been required to create an extra file called `inference.py` which loads the model and transforms the input.
- To query the endpoint with a sample input, we have written the function called `identify_dog`
- We can query with a sample input with the following lines:
	- `test_image = "./test_dogs/labrador.jpg"`
	- `ImageDisplay(test_image)`
	- Where test_image holds the path to the image we want to test with 
	
### Deployed Endpoint
![Deployed Endpoint](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/endpoint.png)

### Sample Query
![Sample Query](https://github.com/deeptea22/dog-breed-classification-udacity-aws/blob/main/Relevant%20Screenshots/sample_query.png)
