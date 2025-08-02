from models import *
# from models import resnet18, Wong2020Fast, Engstrom2019Robustness, BaseNormModel, Identity
from attacks import MI_FGSM, DI_MI_FGSM, VMI_FGSM, MI_TI_FGSM, SpectrumSimulationAttack
from attacks.AdversarialInput.SAI import SAI_FGSM, SAI_FGSM_increment
from attacks.AdversarialInput.VMI import VMI_Inner_CommonWeakness
from tester import test_transfer_attack_acc
from data import get_NIPS17_loader
# from defenses import Randomization, JPEGCompression, NeuralRepresentationPurifier
import torch

# 需要搞清楚，哪个模型对于哪个数据集是预训练的
# 其实我更关注的是imagenet和cifar10
# *****了解一下其他的数据集，比如NIPS2017以及DI/TI/VMI/SSA的算法

def get_test_models(defend, temp_test_models):
    test_models = []
    for model in temp_test_models:
        now_model = defend(model)
        now_model.eval()
        now_model.requires_grad_(False)
        test_models.append(now_model)
    return test_models


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    origin_train_models = [

        # BaseNormModel(googlenet(pretrained=True).to(device)),

        # BaseNormModel(alexnet(pretrained=True).to(device)),
        # BaseNormModel(vgg16(pretrained=True).to(device)),
        # BaseNormModel(resnet101(pretrained=True).to(device)),
        # BaseNormModel(shufflenet_v2_x0_5(pretrained=True).to(device)),
        BaseNormModel(mobilenet_v3_small(pretrained=True).to(device)),

        # Identity(Salman2020Do_R50().to(device)),  # resnet50
        # Identity(Debenedetti2022Light_XCiT_S12().to(device))  # XCiT-S12
    ]

    train_models = []

    for model in origin_train_models:
        model.eval().to(device)
        model.requires_grad_(False)
        train_models.append(model)

    origin_test_models = [
        BaseNormModel(alexnet(pretrained=True).to(device)),
        BaseNormModel(vgg16(pretrained=True).to(device)),
        BaseNormModel(resnet101(pretrained=True).to(device)),
        BaseNormModel(shufflenet_v2_x0_5(pretrained=True).to(device)),
        BaseNormModel(mobilenet_v3_small(pretrained=True).to(device)),
        Identity(Salman2020Do_R50().to(device)),  # resnet50
        Identity(Debenedetti2022Light_XCiT_S12().to(device))  # XCiT-S12
    ]

    test_models = []

    for model in origin_test_models:
        model.eval().to(device)
        model.requires_grad_(False)
        test_models.append(model)


    # defensers = [Randomization, JPEGCompression, NeuralRepresentationPurifier]
    # defender_test_models = []
    # for defender in defensers:
    #     defender_test_models.extend(get_test_models(defender, test_models))


    # attacker_list = [VMI_FGSM, SAI_FGSM]
    # attacker_list = [MI_FGSM, SAI_FGSM_increment]
    attacker_list = [VMI_Inner_CommonWeakness]
    # attacker_list = [MI_FGSM, DI_MI_FGSM, MI_TI_FGSM, VMI_FGSM, SpectrumSimulationAttack, SAI_FGSM]
    loader = get_NIPS17_loader(batch_size=1)

    avg = []

    for now_attacker in attacker_list:
        now_result = []
        attacker = now_attacker(train_models)
        print(attacker.__class__)
        result = test_transfer_attack_acc(attacker, loader, test_models)
        # result_Defender = test_transfer_attack_acc(attacker, loader, defender_test_models)
        # now_result.extend([list_mean(result[:24]), list_mean(result[24:48]), list_mean(result[48:])])
        # avg.append(now_result)
        # print(now_result)
        print(result)
        # print(result_Defender)

    print(avg)



if __name__ == '__main__':
    run()
