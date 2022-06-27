
from ai4water.backend import get_attributes, torch

LAYERS = {}

LOSSES = {}

OPTIMIZERS = {}

if torch is not None:
    LAYERS |= get_attributes(torch, 'nn', case_sensitive=True)

    LOSSES |= {
        "MSE": torch.nn.MSELoss,
        "mse": torch.nn.MSELoss,
        "CROSSENTROPYLOSS": torch.nn.CrossEntropyLoss,
        "L1Loss": torch.nn.L1Loss,
        "NLLLoss": torch.nn.NLLLoss,
        "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
        "MarginRankingLoss": torch.nn.MarginRankingLoss,
        "TripletMarginLoss": torch.nn.TripletMarginLoss,
        "KLDivLoss": torch.nn.KLDivLoss,
    }

    OPTIMIZERS |= get_attributes(torch, 'optim', case_sensitive=True)
