from term_grouping import *
from collections import Counter
import time
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--savepath',type=str,default=None,help='Path to save data')
    args = parser.parse_args()
    return args


def many_trials(filename, commutativity_type, numtrials):
    H = parseHamiltonian(filename)
    ops = [term[1] for term in H]
    Nq = max([int(op[-1][1:]) for op in ops]) + 1
    print('--------------')
    print(filename)
    results = []
    runtimes = []
    for i in range(numtrials):
        start_time = time.time()
        cliques = genMeasureCircuit(H, Nq, commutativity_type)
        end_time = time.time()
        results += [len(cliques)]
        runtimes += [end_time - start_time]
    return Counter(results), runtimes


def main():

    args = parse_args()

    Hfiles = ['hamiltonians/H2_6-31g_{}_0.7_AS4.txt'.format(e) for e in ['JW','BK','BKSF','BKT','PC']]

    results = []
    for f in Hfiles:
        for comm_type, comm_str in zip([QWCCommutativity, FullCommutativity],['QWC','FULL']):
            ret, times = many_trials(f, comm_type, 100)
            results += [(f,comm_str,ret,times)]

    print('\n\n--------------')
    print('All trials finished')
    with open(args.savepath, 'w') as savefile:
        for r in results:
            print(r[0],r[1],r[2],'{:.3f}'.format(np.mean(r[3])))
            savefile.write('{0} {1} {2} {3:.3f}\n'.format(r[0],r[1],r[2],np.mean(r[3])))


if __name__ == '__main__':
    main()
