import itertools
import argparse
import os
import imp
import sys

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def copy_with_replace_values(file_path, new_file_name, keys, values):
    with open(file_path, 'r') as f:
        data = f.readlines()
        for i, data_line in enumerate(data):
            for key, value in zip(keys, values):
                if f"'{key}'" in data_line and "sweep_params" not in data_line:
                    # Do the actual replacing here
                    if value == 'default':
                        data[i] = f'#{data[i].split(":")[0]}: default\n'
                    else:
                        data[i] = data[i].split(':')[0]
                        data[i] += f': {repr(value)},\n'

        with open(new_file_name, 'w') as w:
            w.writelines(data)

def make_control_hp(ancestral, train_hp_fname):
    
    epoch_choice = 300
    print(f'Using epoch {epoch_choice}!')
    with open(ancestral, 'r') as f:
        data = f.readlines()
        for i, data_line in enumerate(data):
            if f"'learned_cost_model_path'" in data_line:
                # Do the actual replacing here
                    data[i] = data[i].split(':')[0]
                    s = splitall(train_hp_fname)
                    b = s.index('distfunc_training')
                    sp = os.path.join(*s[b:-1])
                    data[i] += f": os.environ['VMPC_EXP'] + '/classifier_control/{sp}/weights/weights_ep{epoch_choice}.pth',\n"
    parts = splitall(ancestral)
    assert 'inventory' in parts
    assert 'inventory' in train_hp_fname 
    idx = parts.index('inventory')
    ntn = splitall(train_hp_fname)
    id2 = ntn.index('inventory') 
    new_file_name = os.path.join(*(parts[:idx] + ntn[id2:-1] + ['hparams.py']))
    if not os.path.exists(os.path.dirname(new_file_name)):
        try:
            os.makedirs(os.path.dirname(new_file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(new_file_name, 'w') as w:
        w.writelines(data)
    return new_file_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ancestral", help="path to the ancestral config file")
    parser.add_argument("control_template", help="path to the ancestral control file")
    parser.add_argument("--jpb", help="jobs per batch call (to share GPUS)", type=int, default=1)
    args = parser.parse_args()

    conf_path = os.path.abspath(args.ancestral)
    conf_module = imp.load_source('conf', args.ancestral)

    sweep_keys = conf_module.sweep_params
    configs = [conf_module.configuration, conf_module.model_config]

    sweep_values = []

    for key in sweep_keys:
        found = False
        for config in configs:
            if key in config:
                sweep_options = config[key]
                assert isinstance(sweep_options, list), 'Hparams should be set to a list of options to sweep over!'
                sweep_values.append(config[key])
                found = True
        assert found, f'Key {key} not found in any configuration!'

    # Cartesian product
    run_commands = []
    created_files = 0
    for value_setting in itertools.product(*sweep_values):
        # value_setting is a length len(sweep_keys) iterable containing each possible hparam setting
        setting_name = ','.join([f'{key}={str(value).split("/")[-1].replace(" ", "")}' for key, value in zip(sweep_keys, value_setting)])

        new_folder_name = os.path.join(os.path.dirname(conf_path), setting_name)
        if not os.path.exists(new_folder_name):
            os.mkdir(new_folder_name)
        new_exp_file = os.path.join(new_folder_name, 'conf.py')
        copy_with_replace_values(conf_path, new_exp_file, sweep_keys, value_setting)
        #run_commands.append(f'sbatch -A co_rail -p savio3_2080ti -t 10:00:00 -N 1 -n 1 --cpus-per-task=2 --gres=gpu:1 --qos rail_2080ti3_normal --wrap "')

        #run_commands.append(f'python classifier_control/train.py --deterministic 1 {new_exp_file};')
        train_command = f'python classifier_control/train.py --deterministic 1 {new_exp_file}'
        new_control_file = make_control_hp(args.control_template, new_exp_file)
        py_cmd = f'python run_control_experiment.py {new_control_file}'
        py_cmd = 'singularity exec --nv -B /global -B /usr/lib64 -B /var/lib/dcv-gl /global/scratch/stephentian/container/railrl_hand_v2 bash -c \'source /global/scratch/stephentian/anaconda3/etc/profile.d/conda.sh;export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/global/home/users/stephentian/.mujoco/mujoco200/bin;conda activate dfn_mjc;' + train_command + ' && ' +  py_cmd + '\''
        run_commands.append(py_cmd)

        print(f'... Create exp file for {setting_name}')
        created_files += 1

    print(f'Created {created_files} experiment directories.')
    batch_command_fname = os.path.join(os.path.dirname(conf_path), 'batch_command.sh')
    with open(batch_command_fname, 'w') as f:
        cmd_ind = 0
        while cmd_ind < len(run_commands):
            num_cmds = 0
            command = f'sbatch -A co_rail --mail-type=END,FAIL --mail-user=stephentian@berkeley.edu -p savio3_2080ti -t 20:00:00 -N 1 -n 1 --cpus-per-task=2 --gres=gpu:1 --qos rail_2080ti3_normal --wrap "'
            while num_cmds < args.jpb and cmd_ind < len(run_commands):
                command += run_commands[cmd_ind] + " & "
                cmd_ind += 1 
                num_cmds += 1
            command += 'wait"\n'
            f.write(command)
        
    # Make file executable
    import stat
    st = os.stat(batch_command_fname)
    os.chmod(batch_command_fname, st.st_mode | stat.S_IEXEC)
    

if __name__ == '__main__':
    main()
