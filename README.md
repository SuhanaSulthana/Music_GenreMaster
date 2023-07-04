# Music_GenreMaster
This is the README file for the project "Music GenreMaster".

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Usage](#usage)



### Project Overview <a name="project-overview"></a>
Music_GenreMaster is an innovative music genre classification system that leverages machine learning algorithms to accurately categorize audio samples into specific genres. This project aims to provide a comprehensive understanding of the distinct characteristics and elements that define each genre, enabling accurate genre predictions for music tracks. By leveraging machine learning techniques and the GTZAN genre collection dataset, we can create a model that accurately predicts the genre of a given music track.

### Dataset <a name="dataset"></a>
This project focuses on classifying music genres using the GTZAN genre collection dataset. The dataset consists of 1000 audio files, each with a duration of 30 seconds. It comprises 10 music genres, with each genre containing 100 audio tracks in .wav format.
The GTZAN genre collection dataset can be accessed [here](https://example.com/dataset).



### Getting Started <a name="getting-started"></a>
To tackle the genre classification problem, we will employ the K-nearest neighbors (KNN) algorithm. KNN is known for its simplicity and effectiveness in classification tasks, making it a suitable choice for this project. By comparing the similarity measures, particularly the distance, between audio tracks, the KNN algorithm will predict the genre of a given track based on its nearest neighbors in the feature space.

The feature extraction process plays a crucial role in music genre classification. In this project, we will extract Mel Frequency Cepstral Coefficients (MFCCs) from the audio tracks. MFCCs are widely used in speech and audio signal processing tasks and provide a compact representation of the spectral characteristics of the audio.. For this project, we will utilize Mel Frequency Cepstral Coefficients (MFCCs), which are widely used in automatic speech and speech recognition studies.

The feature extraction process involves the following steps:

- Dividing the audio signals into smaller frames, typically around 20-40 ms long.
- Identifying the different frequencies present in each frame
- Separating linguistic frequencies from the background noise
- Applying the Discrete Cosine Transform (DCT) to the linguistic frequencies to discard unnecessary noise. This process selects a specific sequence of frequencies that are most likely to contain valuable information.
### Usage <a name="usage"></a>
To use the Music_GenreMaster project, you can clone the repository to your local machine:
```bash
git clone https://github.com/suhanasulthana/Music_GenreMaster.git
```
Note: Remember to cite the GTZAN genre collection dataset appropriately if you use it for research or other purposes.
