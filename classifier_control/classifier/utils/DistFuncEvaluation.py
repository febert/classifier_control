import yaml
import torch
from classifier_control.baseline_costs.image_mse_cost import ImageMseCost
from classifier_control.cem_controllers.brl_learned_cost import BRLLearnedCost

class DistFuncEvaluation():
    def __init__(self, testmodel, testparams):
        if testmodel is ImageMseCost:
            self.model = ImageMseCost()
        else:
            model_path = testparams['classifier_restore_path']
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
            self.model = testmodel(overrideparams).eval()
            self.model.to(torch.device('cuda'))

    def predict(self, inputs):
        return self.model(inputs)



