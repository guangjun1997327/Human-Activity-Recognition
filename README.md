# Human-Activity-Recognition
The main contribution is the application of effective techniques for handling the sequence-to-sequence classification task on time series data in the field of HAR. 

ABSTRACT 
This code presents a study on Human Activity Recognition (HAR) using body-worn sensors. The goal of the project is to develop accurate recognition systems for human activities and postural transitions based on data collected from a smartphone's tri-axial accelerometer and gyroscope. The raw data was preprocessed to by Z-score normalization method to resolve any biases in the data and then Sliding window technique was applied to ensure that all samples had a fixed length. Two recurrent neural network models were implemented and trained on the dataset : Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU).  The models' hyperparameters were optimized to achieve the best results. Grid Search is used to optimize the hyperparameters of the models, resulting in an accuracy of 94.48% for the LSTM model and 93.47% for the GRU model. The performance of the models is evaluated using confusion matrix and visualization methods. The main contribution of this work is the application of effective techniques for handling the sequence-to-sequence classification task on time series data in the field of HAR. 
 
1 Introduction 
Human Activity Recognition (HAR) is a non-trival area of research focused on precisely categorizing human activities from sensory data with a wide range of applications in wearable technology, mobile equipment, rehabilitation, and assisted living systems. [4]Besides, activity recognition also has the potential to bring about significant societal benefits, particularly in real-life applications such as elderly care and healthcare.[1] The widespread use of smart devices and the Internet of Things has made it possible to integrate sensors into wearable devices like cell phones, allowing for continuous, non-intrusive recording of human activity information [2] [3].

2 Input Pipeline 

2.1 HAPT Dataset 
This study utilized data from a experiment that involved 30 participants with ages ranging from 19 to 48. The volunteers performed six different activities including walking, walking up stairs, walking down stairs, sitting, standing, and lying. In addition, there were 6 postural transitions between the three static activities of sitting, standing, and lying. The data was recorded using the body-worn sensor positioned at the waist at a constant rate of 50 Hz. The 3-axial linear acceleration and 3-axial angular velocity were recorded. The dataset was further divided into three parts - training, validation, and test - with a 7:2:1 ratio.

2.2 Data Preprocessing 
The preprocessing process began by removing all unlabeled data from the dataset. Then, The remaining data underwent z-score normalization to achieve the zero mean and unit variance. The sliding window technique was applied to ensure equal length sequences in the data, and it can also augument the dataset. Two parameters, the window size and window shift, were set to 250 and 125.

3 Hyperparameter Optimization 
The hyperparameters for the models, such as the number of rnn neurons, number of rnn  layers, total step, number of dense units and drop out rate, were optimized to improve model accuracy. The optimization results for the LSTM model can be found in Table 1, and for the GRU model, the results can be seen in Table 2. The tuning process was performed using the wandb tool and Grid search method.

4 Visualization
We display the outcome in Figure 1 for a whole sequence from the test set to gain several advantages of visualization, such as better understanding of the results and improved interpretability. Visualizing the results can also help in identifying any trends or patterns in the data, making it easier to spot any potential errors or areas for improvement.[5] For our results,there is a liimtation of prediction for postural transition.

5 Results
5.1 LSTM Result 
The LSTM model achieved an test accuracy of 94.48%. The confusion matrix and the normalized confusion matrix for the test dataset are depicted in Figures 1 and 2.
5.2 GRU Result 
The GRU model achieved an test accuracy of 93.47%. The confusion matrix and the normalized confusion matrix for the test data are depicted in Figure 3 and 4.



6, Conclusion and future work
In this project, valuable and effective machine learning techniques for the sequence-to-sequence classification task with time-series data were presented. The performance of our models was evaluated through visualization and confusion matrix methods, which helped us to identify any potential errors or areas for improvement. Our models with optimized hyperparameters achieved outstanding results on sensor-based human activity recognition datasets and demonstrated good performance. Although the results were promising, further work is needed to improve the performance of postural transitions. 
