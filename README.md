# Handwritten Urdu Character Recognition in Machine Learning using Scikit-learn

## 1. Introduction
Many localized languages struggle to reap the beneﬁts of recent advancements in character recognition systems due to the lack of substantial amount of labeled training data. This is due to the difﬁculty in generating large amounts of labeled data for such languages and inability of machine learning techniques to properly learn from small number of training samples. This problem is solved by introducing a technique of generating new training samples from ground-up with realistic augmentations which reﬂect actual variations that are present in human hand writing, by adding random controlled noise to their corresponding instantiation parameters. The results with a mere **40 handwritten papers** and about **280 samples of each urdu character per class** (i.e. **Alif**, **Bay**, **Jeem** and **Daal**) surpass existing character recognition results in Urdu while achieving. This system is useful in character recognition for localized languages that lack much labeled training data and even in other related more general contexts such as object recognition. A strategey has also been developed to implement a simple pipeline which can be used by any machine learning enthusiast. 

## 2. The Pipeline:
The generic flow of this simple pipeline is illustrated below:

<br><img width="280" alt="image" src="https://user-images.githubusercontent.com/61377755/209229446-2ffd52a2-f174-42e2-8014-5db5aacb358a.png">
