# GenAI for Software Development Assignment 1

Benjamin Tremblay, Rowan Miller

- [1 Introduction](#1-introduction)
- [2 Setup](#2-setup)
- [3 Run Model](#3-run-model)
- [4 Report](#4-report)

## **1. Introduction**

We have fine-tuned a Transformer model so that, when given a callcenter conversation while masking the agent's response, the model will predict the hidden response. The code uses the HuggingFace T5 Transformer model maker. To train the model, we mask the agent response statements for training and validation datasets. We then used the a tokenizer form the HuggingFace Tokenizers to tokenize the datasets, adding the <MASK> token to take the place of the if statements and then resizing the model. After training the model, we used the train dataset (after masking) to evaluate using the ROUGE Evaluation metric.  

## **2. Setup**

This project is implemented in **Python 3** and is compatible with **macOS, Linux, and Windows**.

Clone the repository to your workspace:

```shell
~ $ git clone https://github.com/bentrem38/Agent-Reponse-model
```

Navigate into the repository:

```shell
~ $ cd Agent-Reponse-model
~/If-Predicter $
```

## **4. Run Model**

To finetune, test, and evaluate the transformer model, simply run the following:

`python ag_resp.py`

The file `results.csv` will display the results of the model's evaluation on selected tests after running the program. 

## 5. Report

Our overall report is available in the file Final_Project Report.pdf.
