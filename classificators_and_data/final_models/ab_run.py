from ExperimentManager import ExperimentManager
from ExperimentManager import SplitStrategy
from neural_networks.dgcnn_final_model import DcgnnFinalModel
from neural_networks.lstm_final_model import LstmFinalModel
from neural_networks.fbcnet_final_model import FbcnetFinalModel


# models = [DcgnnFinalModel(), LstmFinalModel("gpu"), FbcnetFinalModel("gpu")]
models = [FbcnetFinalModel("gpu")]
split_strategies = [
    SplitStrategy.RANDOM42,
    SplitStrategy.RANDOM2137,
    SplitStrategy.SMALLTEST42,
    SplitStrategy.SUBJECTBASED42,
]

em = ExperimentManager(models, split_strategies)
em.run()
