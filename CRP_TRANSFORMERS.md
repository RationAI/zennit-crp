# Project: Transformer CRP

This is an initial, high-level assignment for research project exploring concept-based XAI approaches for transformer models. The high-level descriptions are interlaced with occasional implementation details, which are necessary fro smooth progress.
The basis for the XAI is CRP (Concept Relevance Propagation) - originally developed for CNNs, but we will adapt it to vision transformers. CRP paper provides implementation for CNNs, papers exist that adapt it to language transformers. The goal is to implement CRP for vision transformers, and then explore how the explanations can be used for debugging and improving the model.

## key concepts:
 - explainability happens through  LRP (Layer-wise Relevance Propagation) algorithm. LRP propagates differently from gradients, requires special propagation rules and hyperparameters.
 - concepts are defined by example images (or text snippets) that activate the same neurons in the model. (defined by CRP paper), to speed up things, CRP implementation provides means to precompute feature visualisation index.
 - specific NN parts are assumed to be concept detectors. In original CRP paper, these are convolutional filters. In vision transformers, we will explore different options.
 - to assess the relevance of specific concept detector, relevance is aggregated across neurons comprising the concept detector. In original CRP, aggregation (max or sum) happens accross conv output. In vision transformers, we will explore different options:
    - attention head is concept detector: most general, relevance is aggregated across whole head output, all head-corresponding token dimensions in all tokens in output sequence, the full triple of weight matrices (query, key, value) is considered as a single unit. (one concept per head)
    - in each head, key, query and value matrices are separate concept detectors: relevance is tracked separately for each of the three matrices, detector relevance is computed by aggregating over all head-corresponding token dimensions in neurons immediately after the matrix multiplication (before attention between the key/query embeddings is computed and used for combining value embeddings) and over the whole sequence. (three concepts per head)
    - individual columns/rows (not sure which one matches the computation flow correctly, investigate exact indexing) of the k/q/v matrices are concept detectors: each token dimension in the output sequence is tracked separately, detector relevance is computed by aggregating just single token dimension across the whole sequence. (as many concepts per head as there are token dimensions corresponding to that head in the output sequence)
 The laid out approaches assume that attention head output tokens are concatenated. If they are averaged, the definition of concept detectors and relevance aggregation will need to be adapted accordingly.
 - LRP propagation can be conditioned on specific concepts. In original paper, relevance is propagated only through 
 select filters. This allows them to visualise heatmaps of specific concepts,detected by those filters. This needs to be adapted to the variosu concept definitions we will explore for vision transformers.

 ## source papers:

- original CRP paper: https://arxiv.org/abs/2206.03208
- CRP for language transformers (LXT): https://arxiv.org/abs/2402.05602
- additional improvemepts on LRP in vision transformers: https://openreview.net/pdf?id=bZ0MXXoldX

 ## implementations

- zennit-crp: our implementation of CRP for vision transformers, based on the original CRP paper and the LXT paper, proof of concept with single concept definition (attention head as concept detector).
- original LXT: https://github.com/rachtibat/LRP-eXplains-Transformers

 ## goals/tasks:

- implement all concpets definitions for vision transformers and audit the current implementation to make sure it is correct and matches the original CRP paper and the LXT paper theory.
- implement example visualisations. Each concept definition should be visualised on a few examples. Download compatible ViT models and datasets, organise the data and code for visualisation. Prepare comparative visualisations of the different concept definitions with default LRP hyperparameters suggested in the original CRP paper and the LXT papers.
- research possible ways to compare and measure the usability/fidelity/stability/etc of the different concept definitions and visualisations. Implement some of these measures and apply them to the different concept definitions and visualisations. If possible, combare them to a suitable baseline (e.g. gradient-based visualisations, occlusion-based visualisations, random concept definitions, etc). The goal here is to have quantitative proof of the usefulness and comparative performance.

## keep in mind

- do not reinvent the wheel. Familiarize with zennit-crp first. zennit-crp already implements basic primitives for aggregation of relevance, builds on top of the LRP implementation in zennit, and provides a proof of concept for one of the concept definitions. Use it as a basis and build on top of it, rather than implementing everything from scratch.
- pay attention to the level of abstractions you are implementing in, use the most suitable base class for the concept definition, additional rules, Canonizers, etc.
- keep the code modular and reusable, so that it can be easily adapted to different concept definitions and different models.

## implementation plan:

- work as an orchestrator. Spawn subagents for independent tasks, but keep an overview of the whole project and make sure all the pieces fit together. 
- start with researching the original CRP paper and the LXT paper, understand the theory and the general concepts. 
- prepare concrete implementation steps, design the code structure, decide on the level of abstraction for the different components (concept definitions, relevance aggregation, LRP propagation rules, etc).
- split the implementation into smaller tasks, implement them one by one
- prepare tests for most important components, especially for the relevance aggregation and LRP propagation rules, to make sure they are implemented correctly and match the theory.
- no cut corners, it is crucial to have a correct implementation, even if it takes more time. 
- if unsure how to proceed, research the literature, look for similar implementations, ask for help if needed, but do not just guess or implement something that does not match the theory.
- avoid arbitrary fallbacks, if something is not working, try to understand why and fix it, rather than just implementing a workaround that may not be correct or may not work in the long run.
- if really unsure, or the incorrect decision may cause significant issues, ask for help.
- if a problem is non-blocking, defer into a log and move on to the next task.
- track current state in CURRENT_STATE.md, keep it up to date, and use it to keep an overview of the project and the current progress.
- keep planned/pending implementation steps in FUTURE_STATE.md (milestones + cross-cutting items), update it as work lands or gets re-scoped.
- after each implementation phase do a sanity checks, try to come up with relevant tests and verify the added changes fit the rest.
- continue until all the planned implementation steps are done or only the blocking problems remain.
