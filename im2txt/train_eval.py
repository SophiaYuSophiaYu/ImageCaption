"""
Train and Evaluate the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train_input_file_pattern', type=str, default='')
    parser.add_argument('--pretrained_model_checkpoint_file', type=str)
    parser.add_argument('--train_dir', type=str, default='')
    parser.add_argument('--train_CNN', type=bool, default=False)
    parser.add_argument('--number_of_steps', type=int, default=300000)
    parser.add_argument('--log_every_n_steps', type=int, default=1)

    # eval
    parser.add_argument('--eval_input_file_pattern', type=str, default='')
    # parser.add_argument('--eval_checkpoint_dir', type=str, default='') 直接用train_dir即可
    parser.add_argument('--eval_dir', type=str, default='')
    parser.add_argument('--eval_interval_secs', type=int, default=600)
    parser.add_argument('--num_eval_examples', type=int, default=10132)
    parser.add_argument('--min_global_step', type=int, default=5000)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

train_cmd = 'python ./train.py ' \
            '--input_file_pattern={input_file_pattern} ' \
            '--inception_checkpoint_file={inception_checkpoint_file} ' \
            '--train_dir={train_dir} ' \
            '--train_inception={train_inception} ' \
            '--number_of_steps={number_of_steps} ' \
            '--log_every_n_steps={log_every_n_steps} '
eval_cmd = 'python ./evalute.py ' \
           '--input_file_pattern={input_file_pattern} ' \
           '--checkpoint_dir={checkpoint_dir} ' \
           '--eval_dir={eval_dir} ' \
           '--eval_interval_secs={eval_interval_secs}  ' \
           '--num_eval_examples={num_eval_examples}  ' \
           '--min_global_step={min_global_step}'

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    step_per_epoch = 50000 // 32

    if FLAGS.pretrained_model_checkpoint_file:
        ckpt = ' --inception_checkpoint_file=' + FLAGS.pretrained_model_checkpoint_file
    else:
        ckpt = ''
    for i in range(30):
        steps = int(step_per_epoch * (i + 1))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'input_file_pattern': FLAGS.train_input_file_pattern,
                                         'inception_checkpoint_file': FLAGS.pretrained_model_checkpoint_file,
                                         'train_dir': FLAGS.train_dir,
                                         'train_inception': FLAGS.train_CNN,
                                         'number_of_steps': FLAGS.number_of_steps,
                                         'log_every_n_steps': FLAGS.log_every_n_steps}) + ckpt)
        for l in p:
            print(l.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'input_file_pattern': FLAGS.eval_input_file_pattern,
                                        'checkpoint_dir': FLAGS.train_dir,
                                        'eval_dir': FLAGS.eval_dir,
                                        'eval_interval_secs': FLAGS. eval_interval_secs,
                                        'num_eval_examples': FLAGS.num_eval_examples,
                                        'min_global_step': FLAGS.min_global_step}))
        for l in p:
            print(l.strip())

