# SpaCCS: A Spatial Multi-Omics Integration Framework

SpaCCS is a unified computational framework for the integration of spatial
multi-omics data, which leverages graph neural networks and combines cross-modal
and intra-modal contrastive learning strategies.

## Overview

SpaCCS is a contrastive learning framework for integrative analysis of spatial
multi-omics data. Each modality is independently encoded using a graph
convolutional network that captures both molecular expression patterns and local
spatial context.

Within each modality, a similarity-aware negative sampling strategy is employed
to guide contrastive learning, encouraging neighboring spots to be closer in the
embedding space while pushing non-neighboring spots apart, thereby preserving
spatial continuity.

Across modalities, cross-modal contrastive learning aligns representations of
corresponding spots to bridge distributional discrepancies. An attention
mechanism is then used to adaptively fuse modality-specific embeddings.

The model is jointly optimized with a composite loss that integrates intra-modal
contrastive, cross-modal contrastive, and reconstruction objectives. This yields
a unified representation that supports downstream tasks such as spatial domain
identification.

## Usage

```bash
python Simulation.py
python "Mouse Embryo E10.py"
```

## Download All Datasets Used in SpaCCS

The datasets used in this paper can be downloaded from the following websites.

### (1) Simulated Dataset

Generated based on established protocols for simulating spatial multi-omics data.
The code and protocols are available at:

<https://github.com/JinmiaoChenLab/SpatialGlue>

### (2) MISAR-seq (Mouse Embryonic Brain E15.5)

Generated using the MISAR-seq technology. The preprocessed data is available at:

<https://github.com/Gaocongqiang/SpaMI>

### (3) Spatial ATAC-RNA-seq (Mouse Brain P22)

Generated using the Spatial ATAC-RNA-seq technology. The data can be accessed at:

<https://brain-spatial-omics.cells.ucsc.edu>

### (4) Stereo-CITE-seq (Mouse Embryo E10)

This dataset is part of the spatial genomics datasets used in the spaVAE study,
available at:

<https://figshare.com/articles/dataset/Spatial_genomics_datasets/21623148/5>
