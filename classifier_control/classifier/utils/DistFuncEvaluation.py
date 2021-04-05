import yaml
import numpy as np
import torch
from classifier_control.baseline_costs.image_mse_cost import ImageMseCost


class DistFuncEvaluation():
    def __init__(self, testmodel, testparams):
        self.models = []

        if testmodel is ImageMseCost:
            self.models.append(ImageMseCost())
        else:
            model_path = testparams['classifier_restore_path']
            model_paths = [model_path]

            for model_path in model_paths:
                testparams['classifier_restore_path'] = model_path
                if model_path is not None:
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

        self.model = self.models[0]

    def predict(self, inputs):
        scores = [model(inputs) for model in self.models]
        return np.min(np.stack(scores), axis=0)



