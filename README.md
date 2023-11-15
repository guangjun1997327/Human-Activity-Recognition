# Human Activity Recognition (HAR)

## Abstract
This repository contains the implementation of advanced machine learning techniques for Human Activity Recognition (HAR) using body-worn sensors. The goal is to accurately classify human activities and postural transitions from tri-axial accelerometer and gyroscope data collected from smartphones. This project presents optimized recurrent neural network models, specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), achieving high accuracy in activity classification.

## Introduction
HAR is a significant research area with applications in wearable technology, mobile health, and assisted living. With the proliferation of smart devices, integrating sensors into everyday wearables has become more feasible, providing valuable data for monitoring and understanding human activities.

## Input Pipeline

### HAPT Dataset
The dataset includes activities recorded from 30 participants, performing activities like walking, stair climbing, and postural transitions. Data was sampled at 50 Hz and was subsequently divided into training, validation, and test sets.

### Data Preprocessing
Unlabeled data were discarded, and the remaining data was normalized using the Z-score method. A sliding window approach was applied to ensure fixed-length sequences for model input.

## Hyperparameter Optimization
Model hyperparameters were fine-tuned using Grid Search and the Weights & Biases tool, improving model accuracy. The optimization process is detailed in the associated tables for both LSTM and GRU models.

## Visualization
Result visualizations provide insights into the model's performance and help in identifying areas for improvement. Figures demonstrating the confusion matrices for each model are included.

## Results
The LSTM model reached an accuracy of 94.48%, while the GRU model achieved 93.47%. Confusion matrices are provided for both.

## Conclusion and Future Work
The project successfully applies sequence-to-sequence classification to time-series sensor data. Future work will aim to enhance the recognition of postural transitions.

## How to Use This Repository
- Clone the repo: `git clone https://github.com/guangjun1997327/Human-Activity-Recognition.git`
- Navigate to the repository: `cd Human-Activity-Recognition`
- Install requirements: `pip install -r requirements.txt`
- Run the preprocessing script: `python preprocessing.py`
- Train the models: `python train.py --rnn [lstm|gru]`
- Evaluate the models: `python evaluate.py --rnn [lstm|gru]`

## Contributing
We welcome contributions! Please read `CONTRIBUTING.md` for how to contribute to this project.

## License
This project is licensed under [LICENSE NAME] - see the `LICENSE.md` file for details.


Thank you for your interest in our Human Activity Recognition project!
