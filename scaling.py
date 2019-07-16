from term_grouping import *
import glob


def print_cliques_loc(filelist):
    for f in filelist:
        print('--------')
        print(f)
        H = parseHamiltonian(f)

        ops = [term[1] for term in H]
        Nq = max([int(op[-1][1:]) for op in ops]) + 1
        print('{} qubits'.format(Nq))

        print('{} total terms\n'.format(len(H)))

        for commutativity_type, type_str in zip([QWCCommutativity, FullCommutativity],['QWC','FULL']):
            for cover_method, cover_str in zip([BronKerbosch, NetworkX_approximate_clique_cover], ['BronKerbosch', 'BoppanaHalldorsson']):
                print(type_str + 'Commutation:')
                print(cover_str + ' algorithm:')
                cliques = genMeasureCircuit(H, Nq, commutativity_type, clique_cover_method=cover_method)
                print()


def main():
    # get hamiltonians
    hfiles_temp = glob.glob('hamiltonians/*')
    hfiles = [h for h in hfiles_temp if not 'taper' in h]

    print_cliques_loc(hfiles)


if __name__ == '__main__':
    main()

