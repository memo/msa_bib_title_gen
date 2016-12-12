Generate fun titles (or phrases) that sound scientific from a [bibtex](http://www.bibtex.org/) file

Unabbreviates a given word or phrase using words from the titles of a given bibtex file.
Useful for generating fun titles for papers or other shennigans.

e.g.

JUNKHEAD:

- Joint Unconstrained Networks Kinematic Handwriting Evaluating Artistic Density
- Joint Unreasonable Neocortex Knowledge Hierarchical Ecosystems Agent Detection
- Joint Understanding Novel Kinematic Hand Embeddings Arm Database
- Joint Unsupervised Neural Kinematics Human Evaluation Algorithmic Design
 
BUTTFACE:

- Bayesian Unique Theorem Taxonomy Formal Autonomous Convolutional Emotion
- Biological Unsupervised Transfer Tracking Fictional Algorithmic Coordination Experience
- Basic Unconstrained Trajectory Tutorial Fitting Animation Calligraphic Embedding 
- Backpropagation Unreasonable Temporal Translation Framework Analysis Connectionist Environment
 
(results aren't always amazing, manual mix-and-matching from multiple samples gives best results)    
    

#USAGE:

Minimum parameters:

>msa_bib_title_gen --phrase [phrase to unabbreviate] --bib_path [path to bibtex file] 

In case you don't have a bibtex file, I've provided mine (msa.bib). Papers on mostly AI, philosophy, cognitive science, robotics, graphics etc.

for help run:

>msa_bib_title_gen --help


#REQUIREMENTS:

- python 2.7 (probably runs on 3.x too, maybe with minimal changes)
- numpy 
- bibtexparser
     
    
#TODO:
- add support for text files (instead of bibtex)
- add support for arxiv (scrape arxiv for titles, by category, CS, Physics, Astronomy etc)
- error handling
- use end probability
- make online version

