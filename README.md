# Introductory Experiments in GNN
M. Saeed Abrishami


## Datasets
### AIFB
From <cite>[A Collection of Benchmark Datasets for Systematic Evaluations of Machine Learning on the Semantic Web \[1\]][1]</cite>:  The AIFB dataset describes the AIFB research institute (*KIT Institut für **A**ngewandte **I**nformatik und **F**ormale **B**eschreibungsverfahren*) in terms of its staff, research group, and publications. In <cite>[Kernel Methods for Mining Instance Data in Ontologies \[2\]][2]</cite> the dataset was first used to predict the affiliation (i.e., research group) for people in the dataset. The dataset contains 178 members of a research group, however the smallest group contains only 4 people, which is removed from the dataset, leaving 4 classes. Also, we remove the employs relation, which is the inverse of the affiliation relation. 

From [2]: the known AIFB dataset is part of **SWRC ontology dataset** \[3\]. The SWRC ontology generically models key entities relevant for typical research communities and the relations between them. The current version of the ontology (2006) comprises a total of 53 concepts in a taxonomy and 42 object properties, 20 of which are participating in 10 pairs of inverse object properties. All entities are enriched with additional annotation information. SWRC comprises at total of **six top level concepts**, namely the **Person, Publication, Event, Organization, Topic and Project concepts** as shown in Figure 1:


The **November 2006 version of the AIFB metadata** comprises an additional set of 2547 instances.
- **1058 of these can be deduced to belong to the Person class, 178 of which have an affiliation to one of the institute’s groups (the others are external co-authors)** with 78 of these being currently employed research staff. 
- *1232* instances instantiate the Publication class
- *146* instances instantiate the ResearchTopic class 
- *146* instances instantiate the Project class. 
- The instances are connected by a total of 15883 object property axioms and participate in a total of 8705 datatype properties.

[1]: https://madoc.bib.uni-mannheim.de/41308/1/Ristoski_Datasets.pdf
[2]: http://iswc2007.semanticweb.org/papers/057.pdf
[3]: https://link.springer.com/chapter/10.1007/11595014_22
<cite>[\[3\] The SWRC Ontology – Semantic Web for Research Communities][3]</cite>

## How to use torch_geometric datasets:

