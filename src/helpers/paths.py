import os

_dir_path = os.path.dirname(os.path.realpath(__file__))
_project_root = os.path.abspath(f'{_dir_path}/../..').replace('\\', '/')

data_path = f'{_project_root}/data'
gensim_data_path = f'{_project_root}/gensim-data'
output_path = f'{_project_root}/output'

eval_script = f'{_project_root}/eval_scripts/evalF1.pl'
