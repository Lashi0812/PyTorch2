[Original Paper --Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)     


# Introduction
**Background information:** The 2012 ImageNet competition winning entry by Krizhevsky et al. ("AlexNet") has been successfully applied to various computer vision tasks. The quality of network architectures has improved by utilizing deeper and wider networks, with VGGNet having architectural simplicity but high computational cost and GoogLeNet having low computational cost but complex architecture.

**Problem statement:** ==The complexity of GoogLeNet makes it difficult to adapt to new use-cases while maintaining efficiency==, and there is a lack of clear explanation about the contributing factors to its design decisions.

**Purpose of the study:**  ==To describe general principles and optimization ideas for scaling up convolution networks== in efficient ways, with a focus on Inception-style networks.

**Research questions or hypotheses:** How can we scale up convolution networks in efficient ways, with a focus on Inception-style networks?

**Outline of the paper:** The paper starts with a description of general principles and optimization ideas for scaling up convolution networks, and focuses on Inception-style networks by observing the principles in their flexible structure. The paper ends with a caution on observing guiding principles to maintain the quality of models.

# General Design Principal
## Large-scale experimentation with convolutional networks
- Avoid representational bottlenecks, especially early in the network
- The representation size should gently decrease from inputs to outputs
- Higher dimensional representations are easier to process locally within a network
- Spatial aggregation can be done over lower dimensional embeddings
- Balance the width and depth of the network for optimal performance
## Representational Bottlenecks
- Avoid extreme compression in the network
- Information content cannot be assessed merely by dimensionality
- Dimensionality provides a rough estimate of information content
## Representation Size
- Gently decrease from inputs to outputs
- Theoretically, information content cannot be assessed by dimensionality alone
## Higher Dimensional Representations
- Easier to process locally within a network
- Results in faster training of networks
## Spatial Aggregation
- Can be done over lower dimensional embeddings
- Hypothesized that strong correlation between adjacent units leads to less loss of information during dimension reduction
- Dimension reduction can promote faster learning
## Balancing Network Width and Depth
- Optimal performance reached by balancing the number of filters per stage and depth
- Increasing both width and depth can contribute to higher quality networks
- Computational budget should be distributed in a balanced way between depth and width


For Factorization refer the notebook. 






