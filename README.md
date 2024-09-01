# QAD-FC
A Fact-Checking framework based on Question-Answering Decomposition

## Instalation
- Create a conda environment 
```
conda create -n qad-fc
conda activate qad-fc
```
- Install pytorch
```
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```
- Install flash-attention
```
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn --no-build-isolation
```
- Install the packages in the `requirements.txt` file
```
pip install -r requirements.txt
```
## Experiment
- Claim Verification 
    - Baseline `python models/5wqa/answer_baseline_qa.py`
    - Our method `python models/5wqa/answer_qaD_infoRE_rels.py`
- Q&A Decomposition and Evidence Reconstruction (Optional)
    - Question & Answer Generation `python models/qa_generate.py`
    - Relation Extraction `python relation_extract.py`
    - Information Reorganization `python reorganize_evidence.py`