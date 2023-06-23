# ERAv1_Session8

### Problem Statement
1. Use CIFAR-10 Dataset
2. Make this network:
    a. C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    b. Keep the parameter count less than 50000
    c. Try and add one layer to another
    d. Max Epochs is 20
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):
    a. Network with Group Normalization
    b. Network with Layer Normalization
    c. Network with Batch Normalization
4. Share these details
    a. Training accuracy for 3 models
    b. Test accuracy for 3 models
    d. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images. 
5. write an explanatory README file that explains:
    what is your code all about,
    your findings for normalization techniques,
    add all your graphs
    your collection-of-misclassified-images 
    Upload your complete assignment on GitHub and share the link on LMS
 



### Solution
#### Please refer to [S8_notebook.ipynb](/S8_notebook.ipynb) Jupyter Notebook for the experiments and plots.


#### Modules-API developed for future purposes!
#### [model.py](/model.py)
1. This is the main module that contains the code for the Model Definition.
2. This module contains the 3 classes:
    LayerNormModel
    GroupNormModel
    BatchNormModel
3. This module also contains ```get_model()``` function that takes the ```normalization``` argument and creates the CNN model with corresponding Normalization technique.


#### [trainer.py](/trainer.py)
1. This is the helper module that contains the following:
    1. ```Trainer:``` This class contains the train and test methods that can be used for 
    ```LayerNormalization```, ```GroupNormalization``` and ```BatchNormalization```.
    2. Each of the above classes also contain ```get_stats()``` method, to track the loss and accuracy values.
2. ```get_misclassified_images```: This method takes the model and test_loader and generates the mis-classified images.


#### Loss of 3 different Normalization Techniques
![alt text](/images/epochs_vs_loss.png)

#### Accuracy of CNN models with 3 different Normalization Techniques
![alt text](/images/epochs_vs_acc.png)

#### Sample Misclassifications by the CNN model - Batch Normalization
![alt text](/images/bn_misclassification.png)

#### Sample Misclassifications by the CNN model - Layer Normalization
![alt text](/images/ln_misclassification.png)

#### Sample Misclassification by the CNN model - Group Normalization
![alt text](/images/gn_misclassification.png)


#### Findings:
```Generally, CIFAR-10 dataset requires good epochs > 20.```
1. BatchNormalization, seems to have greater convergence among other normalization techniques. But the loss function is not completely smooth.
2. Group Normalization: After having trained for 20 epochs (3-4 times), the common observation is that the starting error(rate) is minimum. Seems bit unstable (both loss and accuracy, seems to raise/fall rapidly)
3. Layer Normalization: Has the smooth loss curve among the three. Also, convergence rate is higher.
