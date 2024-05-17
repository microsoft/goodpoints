'''
Script to launch multiple wandb agents in a tmux session.
'''

import argparse
import subprocess
import os
import sys
from pathlib import Path
import re


if __name__ == '__main__':
    env_name = os.environ['CONDA_DEFAULT_ENV']

    parser = argparse.ArgumentParser()
    parser.add_argument('--sess', type=str, required=True)
    parser.add_argument('--cmd', type=str)
    parser.add_argument('--sweep_yml', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--gpus', type=str, nargs='+', required=True)
    parser.add_argument('--procs', type=str, nargs='+')

    args = parser.parse_args()

    if args.sweep_yml is not None:
        assert(args.project is not None)
        cp = subprocess.run(f'wandb sweep --project={args.project} '
                            f'{args.sweep_yml}',
                            capture_output=True, shell=True)
        s = cp.stderr.decode('utf-8')
        print(s)
        m = re.search('wandb agent (\w+/\w+/\w+)', s)
        agent_cmd = m.group(0)
    else:
        assert(args.cmd is not None)
        agent_cmd = args.cmd

    cp = subprocess.run('tmux new-session -d -s {}'.format(args.sess),
                        shell=True, capture_output=True)
    if 'duplicate session' in cp.stderr.decode('utf-8'):
        print(f'Abort because session {args.sess} exists!')
        sys.exit(1)
    print('Created new tmux session {}.'.format(args.sess))
    for i, gpu in enumerate(args.gpus):
        cmd = 'conda activate {};'.format(env_name)
        if gpu != -1:
            cmd += f'CUDA_VISIBLE_DEVICES={gpu} '
        if args.procs is not None:
            proc = args.procs[i]
            cmd += f'taskset -c {proc} '
        cmd += agent_cmd
        window_name = f'agent_{i}'
        subprocess.run('tmux new-window -d -t {} -n {}'.format(
            args.sess, window_name), shell=True)
        subprocess.run("tmux send-keys -t {}:{}.0 '{}' Enter".format(
            args.sess,
            window_name,
            cmd), shell=True)
        print('Sending the following command to window {}: {}'.format(
            window_name, cmd))
