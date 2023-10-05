# online-data-deletion

This repository is to produce the results from the NeurIPS 2022 paper [Algorithms that Approximate Data Removal: New Results and Limitations](https://proceedings.neurips.cc/paper_files/paper/2022/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html).

This code is heavily based on [Certified Data Removal from Machine Learning Models](https://github.com/facebookresearch/certified-removal)
### Dependencies

torch, torchvision, scikit-learn, pytorch-dp

### Setup

We assume the following project directory structure:

```
<root>/
--> save/
--> final_results/
```

### Training a differential private (DP) feature extractor

Training a (0.1, 1e-5)-differentially private feature extractor for SVHN:

```bash
python train_svhn.py --data-dir <SVHN path> --train-mode private --std 6 --delta 1e-5 --normalize --save-model
```

Extracting features using the differentially private extractor:

```bash
python train_svhn.py --data-dir <SVHN path> --test-mode extract --std 6 --delta 1e-5
```

### Removing data from an MNIST 3 vs. 8 model

Training a removal-enabled binary logistic regression classifier for MNIST 3 vs. 8 and removing 1000 training points:

```bash
python ./scripts/test_removal_<method>.py --data-dir <MNIST path> --verbose --extractor none --dataset MNIST --train-mode binary --std 0.01 --lam 1e-3 --num-steps 100
```

### Removing data from an SVHN 3 vs. 8 model

Training a removal-enabled binary logistic regression classifier for MNIST 3 vs. 8 and removing 1000 training points:

```bash
python ./scripts/test_removal_<method>.py --data-dir <SVHN path> --verbose --extractor none --dataset SVHN --train-mode binary --std 0.01 --lam 1e-3 --num-steps 2500
```

### Removing data from a Warfarin dosage model

Training a removal-enabled binary logistic regression classifier for MNIST 3 vs. 8 and removing 1000 training points:

```bash
python ./scripts/test_removal_<method>_prox.py --data-dir <SVHN path> --verbose --extractor none --dataset SVHN --train-mode binary --std 0.01 --lam 1e-3 --num-steps 1000
```

where the method tag can be filled with exact (retraining), sekhari, IJ (our method).

### Reference

This code builds on code from the following paper:

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. **[Certified Data Removal from Machine Learning Models](https://arxiv.org/pdf/1911.03030.pdf)**. ICML 2020.