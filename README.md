# Article Separation
!!!THIS README FILE IS WIP!!!

> Python modules for different tasks:
> - separating articles in (historical) newspapers or similar documents (`article_separation`)
> - measuring the performance of article separation algorithms (`article_separation_measure` and `as_eval`)
> - utility functions, e.g. for plotting images together with metadata information (`python_util`)

<!-- TOC -->
## Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Main Packages](#main-packages)
	* [article_separation](#article_separation)
	* [article_separation_measure](#article_separation_measure)
	* [as_eval](#as_eval)
	* [python_util](#python_util)
* [Usage](#usage)
* [See Also](#see-also)
<!-- TOC -->

## Introduction
This repository is part of the European Union's Horizon 2020 project [NewsEye](https://www.newseye.eu/) and is mainly 
used for separating articles in (historical) newspapers and similar documents. 

The  purpose  of  the  NewsEye  project  is  to  enable  historians  and  humanities  scholars  to  investigate
a great amount of newspaper collections. The newspaper pages are digitized and are available as scanned images. 
To ensure efficient work, the data processing steps should be as automatic as possible. Generally, newspapers are 
structured into large numbers of articles.  These usually contain a distinct piece of content or describe a certain
topic and can mostly be understood without any context. Newspaper articles are crucial entities for historians and 
humanities scholars who focus on a specific research area and are only interested in articles related to that topic. 
Additionally, some natural language processing applications, like e.g. topic modeling or event detection, rely on a 
logical structuring of the underlying text, to be able to extract meaningful information. For this reason it is 
important to tackle the article separation (AS) task, which tries to form coherent articles, based on previously detected
baselines and their respective text.

In the following image a schematic overview of the overall AS workflow can be found.

![Article Separation Workflow](images/as_workflow.png)

## Installation
The Python modules in this repository are all tested with Python 3.6. The best way to use the modules is by creating
a virtual environment and install the packages given in the `requirements.txt` file.

The packages should work with TensorFlow 1.12 (`pip install tensorflow==1.12`) to TensorFlow 1.14 (`pip install tensorflow==1.14`).

## Main Packages / Usage

All modules work with metadata information stored in the well-established 
[PAGE-XML](https://www.primaresearch.org/tools/PAGELibraries) format as defined by the Prima Research group. Some
modules require the following folder structure, where PAGE-XML files are stored inside a separate page folder and should 
have the same basename as the image.

```
.
+-- file1.jpg
+-- file2.jpg
+-- file3.jpg
+-- page
|	+-- file1.xml
|	+-- file2.xml
|	+-- file3.xml
```

### article_separation
The most important package is `article_separation` where all scripts can be found to run AS-related tasks. A brief 
description of all modules that can be used in this repository is given in the following. A more detailed description of 
the workflow can be found in the official public deliverable D2.7 (*Article separation (c) (final)*). A link to all 
public deliverables is given [here](https://cordis.europa.eu/project/id/770299/results).

#### Separator Detection
This module is used to detect visible vertical and horizontal separators on a newspaper page. To use it a TensorFlow 
model is needed that was trained on an image segmentation task. An example network can be found in `nets/separator_detection_net.pb`. The underlying model we used is the so called 
[ARU-Net](https://arxiv.org/abs/1802.03345) which is a U-Net extended by two key concepts, attention (A) and
depth (residual structures (R)). To run the separator detection use the `run_net_post_processing.py` file like in the 
following example.

```bash
python -u run_net_post_processing.py --path_to_image_list "/path/to/image/list" --path_to_pb "/path/to/separator_detection_graph.pb" --mode "separator" --num_processes N
```

#### Text Block Detection
The current version of this module is divided into two parts and only needs the PAGE-XML files:
1. Cluster the text lines / baselines on a page based on the [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) algorithm.
2. Based on these clusters create text regions with the [Alpha shape](https://en.wikipedia.org/wiki/Alpha_shape)
   algorithm.
   
The corresponding run scripts are `run_baseline_clustering.py` and `run_textregion_detection.py` which can be run as in 
the following example.

```bash
python -u run_baseline_clustering.py --path_to_xml_lst "/path/to/xml/list" --num_threads N
python -u run_textregion_generation.py --path_to_xml_lst "/path/to/xml/list" --num_threads N
```
#### Heading Detection
The heading detection combines a distance transformation for detecting approximate text heights and stroke widths with 
an image segmentation approach that detects headings in an image. An example network can be found in `nets/heading_detection_net.pb`. The results of both approaches are combined in a 
weighted manner where most weight is put on the net output. To run the heading detection use the 
`run_net_post_processing.py` file like in the following example.

```bash
python -u run_net_post_processing.py --path_to_image_list "/path/to/image/list" --path_to_pb "/path/to/heading_detection_graph.pb" --mode "heading" --num_processes N
```
#### Graph Neural Network
This module is used to solve a relation prediction task, i.e. to predict which text blocks belong to the same article. 
Since a Graph Neural Network (GNN) works on graph data, which is enriched with feature information, this first needs to be 
generated. To run the feature generation process, use the `article_separation/gnn/input/feature_generation.py` file 
like in the following example.

```bash
python -u feature_generation.py --pagexml_list "/path/to/xml/list" --num_workers N
```

The graph data for a single PAGE-XML file will be saved in a corresponding json file, and will include feature 
information from prior modules if the PAGE-XML files were updated accordingly. Usually this entails, on node-level, 
position and size of the text blocks, stroke width and height of the contained text, and an indicator whether the text 
block is a heading. Geometric features about the top and bottom baseline in the text block can also be added. On 
edge-level it is indicated whether two text blocks are separated by a horizontal or vertical separator.

Optionally, visual regions can be added, which can later be used by a visual feature extractor (e.g. ARU-Net) to 
integrate visual features.
```bash
--visual_regions True
```
Similarly, text block similarity features based on word vectors can be integrated, if they are available.
```bash
--language "language" --wv_path "path/to/wordvector/file"
```
Lastly, additional (previously generated) features can be added via external json files both for nodes and edges. 
This is currently being used for text block similarties coming from a BERT.
```bash
--external_jsons "path/to/external/json/file"
```

#### Text block clustering
The Graph Neural Network outputs a confidence graph regarding the afore-mentioned relation prediction task. Based on 
these predictions the last step to form articles is a clustering process. This is done jointly using the 
`article_separation/gnn/run_gnn_clustering.py` file like in the following example.

```bash
python -u run_gnn_clustering.py \
  --model_dir "path/to/trained/gnn/model" \
  --eval_list "path/to/json/list" \
  --input_params node_feature_dim=NUM_NODE_FEATURES edge_feature_dim=NUM_EDGE_FEATURES \
  --clustering_method CLUSTERING_METHOD \
```

For this module a trained GNN model is needed and the number of node and edge features needs to be set according to 
the GNN, to correctly build the input pipeline. For clustering algorithms, we currently support a greedy approach 
(greedy), a modified DBSCAN algorithm (dbscan) and a hierarchical clustering method (linkage).

If visual regions were generated in the previous step and the GNN was trained accordingly, i.e. a visual feature 
extractor component was added to the network, additional visual features can be integrated during this process. Note 
that in this case the corresponding image files will be needed.
```bash
--image_input True --assign_visual_features_to_nodes True --assign_visual_features_to_edges False
```
The output of this module will be new PAGE-XML files containing the final clustering results, which represent the found
articles.

-----------

### article_separation_measure
This package contains a method to measure the performance of an AS algorithm. It is based on the baseline detection 
measure that was already used at competition like the *ICDAR 2017 Competition on Baseline Detection* and a description 
of it can be found [here](https://arxiv.org/pdf/1705.03311.pdf). The AS measure was used at the [*ICPR 2020 Competition 
Text Block Segmentation on a NewsEye Dataset*](https://link.springer.com/chapter/10.1007%2F978-3-030-68793-9_30). A more 
detailed description can be found in the public deliverable D2.7.

To run the measure you need a list of hypothesis PAGE-XML files and a list of ground truth PAGE-XML files you want to 
compare to. The run script is given by `run_measure.py` and can be executed as in the following example.

```bash
python -u run_measure.py --path_to_hyp_xml_lst "/path/to/hyp/xml/list" --path_to_gt_xml_lst "/path/to/gt/xml/list"
```

-----------

### as_eval
This is another package for evaluating an AS as described in deliverable `D2.7 v6.0` based on how many splits and merges of partition blocks (e.g. text blocks) are needed to convert the ground truth to the hypothesis.

####   Minimal example run script
-   `minRunEx.py`
-   works on example data in
~~~
../work/
    ├ page/example-[1..?].xml               PAGE-XML with ground truth article separation
    └ clustering/                           PAGE-XML with hypotheses …
        └ method-[1..?]/example-[1..?].xml  … for various methods
~~~

- Simplification: For correct interpretation & labeling, a hypothesis' PAGE-XML parent directory's name must be the method's name.
  (cf. `SepPageCompDict.path2method`)

####  Result interpretation

##### SepPageComparison container
- container for counting results
- collected in `SepPageCompDict`
- … counts number of …
  - `gtNIs` … articles in ground truth
  - `hypNIs` … articles in hypothesis
  - `corrects` … properly separated articles in hypothesis
- walking from the ground truth partition to the the hypothesis partition requires … 
    - `splits` many splittings of partition blocks (increasing their number, thus understood to be ≥0)
    - `merges` many mergings of partition blocks (decreasing their number, thus understood to be ≤0)
- … resulting in a partition distance of `dist` = `splits` -  `merges`
- … and requiring consistency (checked by `SepPageComparison.checkConsistency`)
`gtNIs` + `splits` -  `merges` = `hypNIs`

##### XLSX file
`SepPageComparison` containers are ordered by ascending `dist` firstly and descending `corrects` then. In this sense, one method yielded a better article separation, i.e. _gains a victory_ over the other method. We count such _wins_, where (1) ties are also counted as wins, and (2) each method is also compared against itself.
Note that, due to these (laziness!) conditions, all counters always include victories against oneself, i.e. are the number of samples at least.

The `example` worksheet(s) (in general: named after the dataset(s)) contain(s) results for pairwise comparison, where
- main diagonal entries simply count, hence must be equal to the number of comparisons, thus serving as a plausibility check made possible by conditions (1) and (2);
- off-diagonal entries show the ratio between victories of the row-head method over the column-head method, thus resulting in a "reciprocal symmetric" matrix;
- the first column counts all victories of the row-head method.

The `winner` worksheet contains the overall numbers of all victories of the row-head method in any of the datasets under consideration. In column `B`, this includes all methods under investigation. Then the methods with the fewest victories are removed step-by-step, and the number of victories is computed w.r.t. the reduced set of methods only. This yields the values in the subsequent columns, and it stops when in the last column, only the winner method is compared against itself, which obviously must yield the number of investigated samples as another plausibility check.

-----------

### python_util
This package contains multiple utility functions that are used by the `article_separation` package. The most important 
ones are the PAGE-XML parser (`page.py`) and the PAGE-XML plotter (`plot.py`) to load and save the metadata information 
related to an image.

## See Also
- Main NewsEye GitHub page: https://github.com/newseye
- Transkribus GitHub page: https://github.com/Transkribus
- ARU-Net GitHub page: https://github.com/TobiasGruening/ARU-Net
