# SAI-FGSM implementation
---
We reconsider the generation of adversarial examples as a search problem within the neighborhood of the real example. Inspired by the effectiveness of heuristic search methods in finding global optima, we propose the Simulated Annealing Iterative Fast Gradient Sign Method (SAI-FGSM), which leverages the principle of thermal annealing and the Metropolis–Hastings acceptance criterion to escape local minima and saddle points, thereby significantly increasing the probability of discovering global optima.

<img width="1209" height="807" alt="image" src="https://github.com/user-attachments/assets/6c9a2d22-519e-4212-a327-c99862008f19" />


# Setting
We choose 1000 images belonging to the 1000 categories from ILSVRC 2012 validation set, which are almost correctly classified by all the testing models. 


All the images are resized to 224 $\times$ 224 and the number of iterations is fixed to 10. The temperature drops from 10 to 0.5 for all models and the perturbation $\varepsilon$ is set to 0.03. We compare our approach with MI, DI, TI, VMI, PAM, and two state-of-the-art methods BSR and VMI-CWA. We choose five normally trained models - AlexNet, VGG-16, ResNet-101, ShuffleNet-V2, MobileNet-V3 from TorchVision and two adversarially trained models - ResNet-50, XCiT-S12 from RobustBench. They contain both normal and robust models, which is effective to evaluate the transferability of algorithms to black-box models with attack success rate(ASR).


# Result
<img width="1171" height="1143" alt="image" src="https://github.com/user-attachments/assets/6466d0dc-71b2-4ec0-9919-ea9fba505150" />

<img width="1182" height="853" alt="image" src="https://github.com/user-attachments/assets/b19ebc57-776b-4477-91cd-7f62fd88683d" />



## Install
### Environment
Please create a new conda environment and run:
```bash
pip install requirements.txt
```


### Data
You can run \
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

> attacks: Some attack algorithms. Including MI, DI, TI, VMI, PAM, BSR, VMI-CW etc.      
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

