# TransAdapter
TransAdapter: Swin Transformer for Feature-Centric Unsupervised Domain Adaptation

![](sources/)


## Abstract

> Unsupervised Domain Adaptation (UDA) aims to
leverage labeled data from a source domain to address tasks
in a related, but unlabeled, target domain. This problem is
particularly challenging when there is a significant gap between
the source and target domains. Traditional methods have largely
focused on minimizing this domain gap by learning domain-
invariant feature representations using convolutional neural net-
works (CNNs). However, recent advancements in vision trans-
formers, such as the Swin Transformer, have demonstrated su-
perior performance across various vision tasks. In this work, we
propose a novel UDA approach based on the Swin Transformer,
introducing three key modules to enhance domain adaptation.
First, we develop a Graph Domain Discriminator that plays a
crucial role in domain alignment by capturing pixel-wise corre-
lations through a graph convolutional layer, operating on both
shallow and deep features. This module also calculates entropy
for the query and key attention outputs to better distinguish
between source and target domains. Notably, our model does
not include a task-specific domain alignment module, making it
more versatile for various applications. Second, we present an
Adaptive Double Attention module that simultaneously processes
windows and shifted windows attention to increase long-range
dependency features. An attention reweighting mechanism is em-
ployed to dynamically adjust the contributions of these attentions,
thereby improving feature alignment between domains. Finally,
we introduce Transferable Transform Parameters, where random
Swin Transformer blocks are selectively transformed using our
proposed transform module, enhancing the model’s ability to
generalize across domains.Extensive experiments demonstrate
that our method achieves state-of-the-art performance on several
challenging UDA benchmarks, confirming the effectiveness of our
approach. Moreover, by applying this transformer-based model to
the object detection task, we establish its utility across a range
of other tasks, highlighting its broad applicability in domain
adaptation.
