import torch
import cv2
import numpy as np
import os
import glob
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html, save_imgs, save_gifs_direct, save_imgs_direct, save_img_direct, save_img, save_html_direct
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
from classifier_control.classifier.models.q_function import QFunctionTestTime
from classifier_control.classifier.models.tempdist_regressor import TempdistRegressorTestTime
from classifier_control.classifier.models.tempdist_regressor import TempdistRegressorTestTime
from classifier_control.classifier.models.multiway_tempdist_classifier import TesttimeMultiwayTempdistClassifier
from classifier_control.classifier.models.variants.base_tempdistclassifier_monotonicity import MonotonicityBaseTempDistClassifierTestTime


def load_trajectory(path):
    """
    Load images from folder.
    :return: Numpy array of shape [T, H, W, C], where C is RGB format
    """
    load_success = True
    img_idx = 0
    image_traj = []
    while load_success:
        print(f'Loading from file {path}/im_{img_idx}.jpg')
        img = cv2.imread(f'{path}/im_{img_idx}.jpg', cv2.IMREAD_COLOR)
        img_idx += 1
        if img is None:
            load_success = False
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
            image_traj.append(img)
    return np.stack(image_traj)


def np2torch(img, device):
    """
    Converts images to the [-1...1] range of the model.
    Also, flips channel dimension into correct location per torch standards.
    """
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


def prep_distfn_input(traj):
    """
    :param traj: [T, H, W, C] np array of images
    :return: dict containing 'current_img' and 'goal_img'
    """
    # Here we use the time dimension as the batch dim
    torch_traj = np2torch(traj, torch.device('cuda'))
    goal_img = torch_traj[-1, ...]  # Take last image in traj as goal
    goal_img_tiled = torch.stack(torch_traj.shape[0]*[goal_img])

    input_dict = {
        'current_img': torch_traj,
        'goal_img': goal_img_tiled
    }
    return input_dict


def run_dist_fn(learned_fn_name, model_path, inputs, output_dict, output_folder):
    _batch_size = inputs['goal_img'].shape[0]
    params = {'batch_size': _batch_size, 'classifier_restore_path': model_path}
    eval_model = DistFuncEvaluation(learned_fn_name, params)
    outputs = eval_model.predict(inputs)
    eval_model.model.visualize_test_time(output_dict, np.arange(_batch_size), output_folder)
    return outputs


models_to_run = [
    (QFunctionTestTime, f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/q_function_training/robonet/gamma0.2/weights/weights_ep166.pth'),
    (TempdistRegressorTestTime, f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/tdist_regression/robonet/weights/weights_ep169.pth'),
    (MonotonicityBaseTempDistClassifierTestTime,
     f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/ensem_classifier/robonet/weights/weights_ep199.pth'),
    (TesttimeMultiwayTempdistClassifier,
     f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/tdist_multiway_classification/robonet/weights/weights_ep194.pth'),
]

examples = ['image_pair_{}'.format(i+1) for i in range(5)]

if __name__ == "__main__":

    for example_name in examples:

        output_folder = f'{os.environ["VMPC_EXP"]}/eval_ex_trajs/{example_name}/vis/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        traj = load_trajectory(f'{os.environ["VMPC_EXP"]}/eval_ex_trajs/{example_name}/traj_data/images0')

        output_dict = {}

        dist_fn_inputs = prep_distfn_input(traj)

        output_dict['goal_image'] = save_imgs_direct(output_folder, 'goal_image', len(traj) * [traj[-1]])
        output_dict['traj_image'] = save_imgs_direct(output_folder, 'traj_image', traj)

        for learned_fn_name, model_path in models_to_run:
            output = run_dist_fn(learned_fn_name, model_path, dist_fn_inputs, output_dict, output_folder)
            print(f'Ran {learned_fn_name},  output {output}')

        html_page = fill_template(0, 0, output_dict)
        save_html_direct(f'{output_folder}/summary.html', html_page)

