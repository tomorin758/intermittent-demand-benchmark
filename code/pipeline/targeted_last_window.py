"""Targeted last-window evaluation for the m5 PatchTCN / PatchTST runs.

Why: their stored arrays are missing the last three rolling windows (355 -> 352) and a full
re-evaluation of all 355 windows is prohibitively slow on this box. Here the *original*
pipeline is reused unchanged (run_longExp -> Exp_Main.test) but the test loader is restricted
to the LAST `LASTN` windows, so window 351 of the run is still available for a direct
comparison against the stored array (which has windows 0..351).

usage:  LASTN=4 CUDA_VISIBLE_DEVICES=... python targeted_last_window.py PatchTCN|PatchTST
"""
import os, sys

os.chdir('${PAPER_ROOT}/Projects/PatchTST/PatchTST_supervised')
sys.path.insert(0, '${PAPER_ROOT}/Projects/PatchTST/PatchTST_supervised')

ARCH = {
    'PatchTCN': ['--e_layers', '3', '--n_heads', '4', '--d_model', '32', '--d_ff', '64',
                 '--dropout', '0.2', '--fc_dropout', '0.2', '--head_dropout', '0',
                 '--patch_len', '16', '--stride', '8', '--mspatch_kernels', '3', '5', '7',
                 '--interpatch_kernel_size', '3', '--batch_size', '4', '--learning_rate', '0.0001'],
    'PatchTST': ['--e_layers', '3', '--n_heads', '4', '--d_model', '32', '--d_ff', '64',
                 '--dropout', '0.2', '--fc_dropout', '0.2', '--head_dropout', '0',
                 '--patch_len', '16', '--stride', '8', '--batch_size', '4', '--learning_rate', '0.0001'],
}
model = sys.argv[1]
assert model in ARCH, model

sys.argv = ['run_longExp.py',
            '--random_seed', '2021', '--is_training', '0',
            '--root_path', './dataset/', '--data_path', 'm5.csv',
            '--model_id', 'm5_112_28', '--model', model, '--data', 'custom',
            '--features', 'M', '--target', 'FOODS_3_090__CA_3', '--freq', 'd',
            '--seq_len', '112', '--label_len', '28', '--pred_len', '28',
            '--enc_in', '30490', '--dec_in', '30490', '--c_out', '30490',
            '--des', 'Exp', '--num_workers', '0', '--itr', '1'] + ARCH[model]
sys.argv = [a for a in sys.argv]

import torch                                                    # noqa: E402
from torch.utils.data import Subset, DataLoader                 # noqa: E402
import exp.exp_main as em                                       # noqa: E402

LASTN = int(os.environ.get('LASTN', '4'))
_orig = em.Exp_Main._get_data


def patched(self, flag, *a, **kw):
    data_set, loader = _orig(self, flag, *a, **kw)
    if flag == 'test':
        n = len(data_set); k = min(LASTN, n)
        sub = Subset(data_set, list(range(n - k, n)))
        loader = DataLoader(sub, batch_size=self.args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
        print(f'[targeted] test loader restricted to the last {k} of {n} windows '
              f'(global indices {n-k}..{n-1})', flush=True)
    return data_set, loader


em.Exp_Main._get_data = patched
# run_longExp.py is a top-level script (everything under __main__), so execute it directly
SRC = '${PAPER_ROOT}/Projects/PatchTST/PatchTST_supervised/run_longExp.py'
code = compile(open(SRC).read(), SRC, 'exec')
exec(code, {'__name__': '__main__', '__file__': SRC})
