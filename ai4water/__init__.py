"""
main models are pytorch based and Model.
All tensorflow based models can be implemented purely using Model.
"""

from ai4water.main import Model

try:
    from ai4water.pytorch_models import IMVModel
    from ai4water.pytorch_models import HARHNModel
except AttributeError:
    print(f"\n{10 * '*'}Pytorch models could not be imported {10 * '*'}\n")

try:
    from ai4water.tf_models import InputAttentionModel
    from ai4water.tf_models import DualAttentionModel
except AttributeError:
    print(f"\n{10 * '*'}Tensorflow models could not be imported {10 * '*'}\n")

__version__ = "1.04"
