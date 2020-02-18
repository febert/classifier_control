import torch
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

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


def prep_distfn_input(traj, goal_index=-1):
    """
    :param traj: [T, H, W, C] np array of images
    :return: dict containing 'current_img' and 'goal_img'
    """
    # Here we use the time dimension as the batch dim
    torch_traj = np2torch(traj, torch.device('cuda'))
    goal_img = torch_traj[goal_index, ...]  # Take last image in traj as goal
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
    del eval_model
    return outputs


def plot_single_timestep(data, t, xlim, ylim, title):
    x_data = np.arange(t+1)
    y_data = data[:t+1]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, len(data))
    ax.plot(x_data, y_data)
    ax.grid()
    ax.set(xlabel="time", ylabel="cost/qval", title=title)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def make_gif_over_time(traj, outputs, goal_index=-1, ylim=(-2, 10), fps=3):
    """
    :param traj: Numpy array of shape [T, H, W, C], where C is RGB format
    :param outputs: dictionary containing {model_name: [T, 1]? which contain the overall costs evaluated for a given model
    :return clip: ImageSequenceClip of summary over time
    """

    goal_image = traj[goal_index]
    h, w, _ = goal_image.shape
    goal_h, goal_w = 4*h, 4*w
    goal_image = cv2.resize(goal_image, dsize=(goal_w, goal_h), interpolation=cv2.INTER_CUBIC)
    total_timesteps = len(traj)
    gif_frames = []
    for t in range(0, total_timesteps):
        traj_image = traj[t]
        traj_image = cv2.resize(traj_image, dsize=(goal_w, goal_h), interpolation=cv2.INTER_CUBIC)
        model_images = []
        for model_name in outputs:
            model_output = outputs[model_name]
            if model_name == "Q function":
                plot_ylim = (-2, 2)
            else:
                plot_ylim = ylim
            plot = plot_single_timestep(model_output, t, total_timesteps, plot_ylim, model_name)
            model_images.append(plot)
        model_plots = np.concatenate(model_images, axis=0)
        robot_images = np.concatenate((goal_image, traj_image), axis=0)
        height_difference = model_plots.shape[0] - robot_images.shape[0]
        robot_images = np.pad(robot_images, ((height_difference//2, height_difference//2), (0, 0), (0, 0)), mode='constant')
        tstep_summary = np.concatenate((robot_images, model_plots), axis=1)
        gif_frames.append(tstep_summary)

    clip = ImageSequenceClip(gif_frames, fps=fps)
    return clip

models_to_run = [
    (QFunctionTestTime, f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/q_function_training/robonet/gamma0.2/weights/weights_ep166.pth', 'Q function'),
    (TempdistRegressorTestTime, f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/tdist_regression/robonet/weights/weights_ep169.pth', 'TD Regressor'),
    (MonotonicityBaseTempDistClassifierTestTime,
     f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/ensem_classifier/robonet/weights/weights_ep199.pth', 'Ensemble of classifiers'),
    (TesttimeMultiwayTempdistClassifier,
     f'{os.environ["VMPC_EXP"]}/classifier_control/distfunc_training/tdist_multiway_classification/robonet/weights/weights_ep194.pth', 'Multiway classifier'),

]

examples = ['image_pair_{}'.format(i+1) for i in range(5)]
examples = ['cmu_push_cube', 'cmu_push_tri']
examples = [f'galar_traj{i}' for i in range(13)]
goal_indices = [-1] * 13

if __name__ == "__main__":

    for e_ind, example_name in enumerate(examples):

        output_folder = f'{os.environ["VMPC_EXP"]}/eval_ex_trajs/{example_name}/vis/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        try:
            traj = load_trajectory(f'{os.environ["VMPC_EXP"]}/eval_ex_trajs/{example_name}/traj_data/images0')
        except:
            traj = load_trajectory(f'{os.environ["VMPC_EXP"]}/eval_ex_trajs/{example_name}/images0')

        output_dict = {}

        dist_fn_inputs = prep_distfn_input(traj, goal_indices[e_ind])

        output_dict['goal_image'] = save_imgs_direct(output_folder, 'goal_image', len(traj) * [traj[goal_indices[e_ind]]])
        output_dict['traj_image'] = save_imgs_direct(output_folder, 'traj_image', traj)
        dist_fn_outputs = {}

        for learned_fn_name, model_path, human_name in models_to_run:
            output = run_dist_fn(learned_fn_name, model_path, dist_fn_inputs, output_dict, output_folder)
            print(f'Ran {learned_fn_name},  output {output}')
            dist_fn_outputs[human_name] = output

        html_page = fill_template(0, 0, output_dict)
        save_html_direct(f'{output_folder}/summary.html', html_page)

        clip = make_gif_over_time(traj, dist_fn_outputs, goal_indices[e_ind])
        clip.write_videofile(f'{output_folder}/summary.mp4', fps=3)
        clip.write_gif(f'{output_folder}/summary.gif', fps=3)

