#  XAI_EvidenceExtraction


# Dependencies
* python 3.7
* PyTorch 1.6.0
* Transformers 2.11.

# Data
* HOTPOT QA

# Train & Test
* Train : run_mrc.py --init_weight True --do_train True
* Test : run_mrc.py --init_weight False --do_eval True --checkpoint [saved model global Step]

