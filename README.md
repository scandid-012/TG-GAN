# TG-GAN
Synthetic EEG for Motor Imagery Tasks using Transformer-Guided Generative Adversarial Network
Abstract—Electroencephalogram (EEG) signals are essential
for brain–computer interface (BCI) systems and clinical research.
However, the limited availability of EEG data constrains the
performance of deep learning models. While existing genera
tive models can replicate time-domain patterns, they often fail
to generate complex temporal features. This study introduces
a Transformer-Guided Generative Adversarial Network (TG
GAN), which integrates adversarial learning with self-attention
mechanisms to generate synthetic EEG signals that capture both
temporal and frequency domain features. Experiments conducted
on the BCI Competition IV 2b dataset indicate that TG-GAN
produces realistic signals with statistical properties, temporal
dynamics, and spectral patterns comparable to real data in the
µ (8–13 Hz) band. Additionally, Fr´ echet Distance scores confirm
the distributional similarity between synthetic and real signals.
These findings indicate that TG-GAN is a promising approach
for augmenting EEG datasets and enhancing model training in
BCI applications.
