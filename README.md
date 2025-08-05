# SAI-FGSM implementation
---
We reconsider the generation of adversarial examples as a search problem within the neighborhood of the real example. Inspired by the effectiveness of heuristic search methods in finding global optima, we propose the Simulated Annealing Iterative Fast Gradient Sign Method (SAI-FGSM), which leverages the principle of thermal annealing and the Metropolis–Hastings acceptance criterion to escape local minima and saddle points, thereby significantly increasing the probability of discovering global optima.

<img width="1209" height="807" alt="image" src="https://github.com/user-attachments/assets/6c9a2d22-519e-4212-a327-c99862008f19" />


# Datasets and Metrics
We test the performance of MASK on two standard benchmark datasets: Flickr30k and MSCOCO.
The image-text matching usually includes two sub-tasks in terms of: 1) image annotation: retrieving related texts given images, and 2) image retrieval: retrieving related images given texts. 
The commonly used evaluation criterions are ``R@1", ``R@5" and ``R@10", i.e., recall rates at the top-1, 5 and 10 results. Following existing works, we also use an additional criterion of ``Rs" by summing all the recall rates to evaluate the overall performance.


# Implementation Details
In the multimodal aligned semantic knowledge, we collect all words from the VG dataset and filter out some special characters and rare words, resulting in a total of $K$=12,385 semantic concepts. For each image, we initially employ the pre-trained object detection model Bottom-UP Top-Down  \footnote{https://github.com/MILVLG/bottom-up-attention.pytorch} to extract raw region representations, setting the number of detected regions to $I$=36 and the dimensionality of each region representation to $M$=2048. 
For each word, we obtain its word embedding using the pre-trained word vectors glove-twitter-50 \footnote{https://nlp.stanford.edu/projects/glove/}. 
The batch size is 4096 for the first 200 epochs and 2048 for the next 200 epochs. The trade-off factors $\lambda_1$ and $\lambda_2$ are set to 3. We use the Adam to optimize the loss with a learning rate of 1e-4.



# Result
<img width="1457" height="1060" alt="image" src="https://github.com/user-attachments/assets/2d41e99d-8ff0-4a1e-9b52-afc16eddaf3f" />

## Install
### Environment
Please create a new conda environment and run:
```bash
pip install requirements.txt
```


### Data
**CIFAR10** will be downloaded automatically    
For **PACS** dataset, please refer to ./data/PACS.py for install    
For **NIPS17** dataset, you can run \
```bash
kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set
```
and then put it into ./resources/NIPS17

### Model Checkpoints
All the models in our paper are from torchvision, robustbench, timm library, and the checkpoints will be downloaded automatically.

We also encapsulate some models and defenses in *"./models"* and *"./defenses"*. If you want to attack them, you can download their checkpoints by yourself

---

## Usage

### Code Framework

> attacks: Some attack algorithms. Including VMI, VMI-CW, CW, SAM, etc.      
> data: loader of CIFAR, NIPS17, PACS    
> defenses: Some defenses algorithm    
> experiments: Example codes    
> models: Some pretrained models   
> optimizer: scheduler and optimizer   
> tester: some functions to test accuracy and attack success rate   
> utils: Utilities. Like draw landscape, get time, HRNet, etc.     


### Basic functions

```
tester.test_transfer_attack_acc(attacker:AdversarialInputBase, loader:DataLoader, target_models: List[nn.Module]) \
```

This function aims to get the attack success rate on loader against target models



```
attacker = xxxAttacker(train_models: List[nn.Module])
```
You can initialize attacker like this.

### Examples
Here is an example of testing attack success rate on NIPS17 loader.
```python
from models import resnet18, Wong2020Fast, Engstrom2019Robustness, BaseNormModel, Identity
from attacks import MI_CommonWeakness
attacker = MI_CommonWeakness([
    BaseNormModel(resnet18(pretrained=True)), # model that requires normalization
    Identity(Wong2020Fast(pretrained=True)) # model that do not need normalization
])

from tester import test_transfer_attack_acc
from data import get_NIPS17_loader
test_transfer_attack_acc(attacker, 
                         get_NIPS17_loader(), 
                         [
                             Identity(Wong2020Fast(pretrained=True)), # white box attack
                             Identity(Engstrom2019Robustness(pretrained=True)), # transfer attack
                          ]
                         )
```


For more example codes, please visit *'./experiments'* folder. There are some example codes using our framework to attack and draw landscapes. I believe you can quickly get familiar with our framework via these example codes.


### HRNet
HRNet is a function that aims to reduce memory cost when crafting adversarial examples.

**We haven't implemented the convolution of HRNet. Up to now, HRNet can only help to reduce about 30% of memory cost**

#### Usage
```python
from models import resnet18
from utils import change

model = resnet18()
model = change(model)
```


---

