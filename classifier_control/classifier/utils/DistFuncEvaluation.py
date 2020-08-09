import yaml
import numpy as np
import torch
from classifier_control.baseline_costs.image_mse_cost import ImageMseCost
from classifier_control.cem_controllers.brl_learned_cost import BRLLearnedCost

class DistFuncEvaluation():
    def __init__(self, testmodel, testparams):
        self.models = []
        if testmodel is ImageMseCost:
            self.models.append(ImageMseCost())
        else:
            model_paths = testparams['classifier_restore_path']

            if isinstance(model_paths, list):
                model_paths = [model_paths]
            for model_path in model_paths:
                if model_path is not None and testmodel is not BRLLearnedCost:
                    config_path = '/'.join(str.split(model_path, '/')[:-2]) + '/params.yaml'
                    with open(config_path) as config:
                        overrideparams = yaml.load(config)
                else:
                    overrideparams = dict()
                if 'builder' in overrideparams:
                    overrideparams.pop('builder')
                overrideparams.update(testparams)
                overrideparams['ignore_same_as_default'] = ''  # adding this flag prevents error because of value being equal to default
                model = testmodel(overrideparams).eval()
                model.to(torch.device('cuda'))
                self.models.append(model)

    def predict(self, inputs):
        scores = [model(inputs) for model in self.models]
        return np.max(np.stack(scores), axis=0)


