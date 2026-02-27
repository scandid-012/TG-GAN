# TG-GAN
Motor imagery-based brain-computer interfaces (BCIs) face challenges due to the 
limitation of good-quality EEG data, which makes it difficult to train accurate 
classification models. This thesis presents a Transformer-guided Generative Adversarial 
Network (TG-GAN) to generate realistic motor imagery EEG signals and help overcome 
data shortages in BCI applications. The model uses Transformer attention along with 
adversarial training to learn both short-term and long-term patterns in EEG signals. It 
was trained on data from three EEG channels (C3, Cz, C4), focusing on left- and right
hand motor imagery tasks. A well-organized preprocessing was done including 
segmentation, normalization, and data reshaping ensures the model receives consistent 
and clean input. The generated EEG data was evaluated using different methods, such as 
basic statistics like average value and variation, checking brainwave activity in important 
frequency ranges (alpha, beta and gamma bands), measuring how well signals from 
different channels are related, and using a similarity score called Fréchet Inception 
Distance (FID). The results show that TG-GAN can reproduce key features of motor 
imagery EEG, especially activity in the beta frequency band (13–30 Hz) and lower FID 
for some generated data. However, the generated signals had slightly higher variance, 
lower power at higher frequencies, and stronger-than-normal correlation between 
channels, which are not typically seen in real EEG. The study presents a promising 
generative model for augmenting EEG data in low-data settings. Future work could focus 
on class-specific generation, subject-based modeling, and testing TG-GAN data in real 
BCI classification tasks.
