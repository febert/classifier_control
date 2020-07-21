import itertools
import argparse
import os
import imp


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ancestral", help="path to the ancestral config file")
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
        setting_name = ','.join([f'{key}={str(value).split("/")[-1]}' for key, value in zip(sweep_keys, value_setting)])

        new_folder_name = os.path.join(os.path.dirname(conf_path), setting_name)
        if not os.path.exists(new_folder_name):
            os.mkdir(new_folder_name)
        new_exp_file = os.path.join(new_folder_name, 'conf.py')
        copy_with_replace_values(conf_path, new_exp_file, sweep_keys, value_setting)
        #run_commands.append(f'sbatch -A co_rail -p savio3_2080ti -t 15:00:00 -N 1 -n 1 --cpus-per-task=2 --gres=gpu:1 --qos rail_2080ti3_normal --wrap "')
        run_commands.append(f'python classifier_control/train.py {new_exp_file}')

        print(f'... Create exp file for {setting_name}')
        created_files += 1

    print(f'Created {created_files} experiment directories.')

    with open(os.path.join(os.path.dirname(conf_path), 'batch_command.sh'), 'w') as f:
        cmd_ind = 0
        while cmd_ind < len(run_commands):
            num_cmds = 0
            command = f'sbatch -A co_rail -p savio3_2080ti -t 15:00:00 -N 1 -n 1 --cpus-per-task=4 --gres=gpu:1 --qos rail_2080ti3_normal --wrap "'
            while num_cmds < args.jpb and cmd_ind < len(run_commands):
                command += run_commands[cmd_ind] + " & "
                cmd_ind += 1 
                num_cmds += 1
            command += 'wait"\n'
            f.write(command)


if __name__ == '__main__':
    main()
