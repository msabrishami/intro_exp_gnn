# Introductory Experiments in GNN
M. Saeed Abrishami


## Datasets
### AIFB
From <cite>[A Collection of Benchmark Datasets for Systematic Evaluations of Machine Learning on the Semantic Web \[1\]][1]</cite>:  The AIFB dataset describes the AIFB research institute (*KIT Institut für **A**ngewandte **I**nformatik und **F**ormale **B**eschreibungsverfahren*) in terms of its staff, research group, and publications. In <cite>[Kernel Methods for Mining Instance Data in Ontologies \[2\]][2]</cite> the dataset was first used to predict the affiliation (i.e., research group) for people in the dataset. The dataset contains 178 members of a research group, however the smallest group contains only 4 people, which is removed from the dataset, leaving 4 classes. Also, we remove the employs relation, which is the inverse of the affiliation relation. 

[1]: https://madoc.bib.uni-mannheim.de/41308/1/Ristoski_Datasets.pdf
[2]: http://iswc2007.semanticweb.org/papers/057.pdf

## How to use torch_geometric datasets:

