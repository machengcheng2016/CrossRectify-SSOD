import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, required=True)
parser.add_argument('-m', type=int, required=True)
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-N', type=int, required=True)
parser.add_argument('-fp', type=int, required=True)
parser.add_argument('-fp_right', type=int, required=True)
parser.add_argument('-tp_wrong', type=int, required=True)
parser.add_argument('-A', action="store_true")
parser.add_argument('-B', action="store_true")
args = parser.parse_args()

f = open(args.f)
a = f.readlines()
f.close()

m, n, N = [], [], []
fp, fp_right, tp_wrong = [], [], []
for i in a:
  if args.A and i.startswith('A') and len(i.split()) >= 3 and i.split()[1].isdigit() and i.split()[2].isdigit():
    m.append(float(i.split()[args.m]))
    n.append(float(i.split()[args.n]))
    N.append(float(i.split()[args.N])+1e-12)
    fp.append(float(i.split()[args.fp]))
    fp_right.append(float(i.split()[args.fp_right]))
    tp_wrong.append(float(i.split()[args.tp_wrong]))
  if args.B and i.startswith('B') and len(i.split()) >= 3 and i.split()[1].isdigit() and i.split()[2].isdigit():
    m.append(float(i.split()[args.m]))
    n.append(float(i.split()[args.n]))
    N.append(float(i.split()[args.N])+1e-12)
    fp.append(float(i.split()[args.fp]))
    fp_right.append(float(i.split()[args.fp_right]))
    tp_wrong.append(float(i.split()[args.tp_wrong]))
  if (not args.A) and (not args.B) and len(i.split()) >= 3 and i.split()[1].isdigit() and i.split()[2].isdigit():
    m.append(float(i.split()[args.m]))
    n.append(float(i.split()[args.n]))
    N.append(float(i.split()[args.N])+1e-12)
    fp.append(float(i.split()[args.fp]))
    fp_right.append(float(i.split()[args.fp_right]))
    tp_wrong.append(float(i.split()[args.tp_wrong]))

m, n, N = np.array(m), np.array(n), np.array(N)
fp, fp_right, tp_wrong = np.array(fp), np.array(fp_right), np.array(tp_wrong)

# find FP
fp_i = fp
fp_idx = (fp_i > 0)
fp_right_i = fp_right
# find TP
tp_i = m
tp_idx = (tp_i > 0)
tp_wrong_i = tp_wrong

print("Own TP/P = {:.3f}, Own TP = {:.1f}, Pseudo TP/P = {:.3f}, Pseudo TP = {:.1f}, FP-correct/FP = {:.3f}, FP-correct = {:.1f}, TP-wrong/TP = {:.3f}, TP-wrong = {:.1f}".format((m/N).mean(), m.mean(), (n/N).mean(), n.mean(), (fp_right_i[fp_idx]/fp_i[fp_idx]).mean(), fp_right_i[fp_idx].mean(), (tp_wrong_i[tp_idx]/tp_i[tp_idx]).mean(), tp_wrong_i[tp_idx].mean()))
