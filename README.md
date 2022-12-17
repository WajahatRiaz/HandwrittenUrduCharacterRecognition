# Handwritten Urdu Character Recognition in Machine Learning using Scikit-learn

## 1. Introduction
Many localized languages struggle to reap the beneﬁts of recent advancements in character recognition systems due to the lack of substantial amount of labeled training data. This is due to the difﬁculty in generating large amounts of labeled data for such languages and inability of machine learning techniques to properly learn from small number of training samples. This problem is solved by introducing a technique of generating new training samples from ground-up with realistic augmentations which reﬂect actual variations that are present in human hand writing, by adding random controlled noise to their corresponding instantiation parameters. The results with a mere 40 handwritten papers and about 280 samples of each urdu character per class (i.e. Alif, Bay, Jeem and Daal) surpass existing character recognition results in Urdu while achieving.  This system is useful in character recognition for localized languages that lack much labeled training data and even in other related more general contexts such as object recognition. A strategey has also been developed to implement a simple pipeline which can be used by any machine learning enthusiast. 

## Pipeline:
The generic flow of this simple pipeline is illustrated below:
<img width="497" alt="image" src="https://user-images.githubusercontent.com/61377755/208210826-0b1ef0a9-d1f8-4290-a2f1-d139b0815568.png">
