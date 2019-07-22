import glob
import time
import sys
import argparse
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import pprint
import copy
import networkx as nx
from networkx.algorithms import approximation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--algorithm',type=str,default='BK',
                        help='MIN-CLIQUE-COVER algorithm')
    parser.add_argument('-c','--commutation',type=str,default='QWC',
                        help='type of commutation graph to generate')
    parser.add_argument('-s','--specificH',type=str,default=None,
                        help='Can specify a particular H')
    args = parser.parse_args()
    return args


class CommutativityType(object):
    def gen_comm_graph(term_array):
        raise NotImplementedError


class QWCCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}
        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                commute = True
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        commute = False
                        break
                if commute:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        print('MEASURECIRCUIT: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


class FullCommutativity(CommutativityType):
    def gen_comm_graph(term_array):
        g = {}

        for i, term1 in enumerate(term_array):
            comm_array = []
            for j, term2 in enumerate(term_array):

                if i == j: continue
                non_comm_indices = 0
                for c1, c2 in zip(term1, term2):
                    if c1 == '*': continue
                    if (c1 != c2) and (c2 != '*'):
                        non_comm_indices += 1
                if (non_comm_indices % 2) == 0:
                    comm_array += [''.join(term2)]
            g[''.join(term1)] = comm_array

        print('MEASURECIRCUIT: Generated graph for the Hamiltonian with {} nodes.'.format(len(g)))

        return g


def prune_graph(G,nodes):
    for n in nodes:
        neighbors = G.pop(n)
        for nn in neighbors:
            G[nn].remove(n)


def degeneracy_ordering(graph):
    """
    Produce a degeneracy ordering of the vertices in graph, as outlined in,
    Eppstein et. al. (arXiv:1006.5440)
    """

    # degen_order, will hold the vertex ordering
    degen_order = []
    
    while len(graph) > 0:
        # Populate D, an array containing a list of vertices of degree i at D[i]
        D = []
        for node in graph.keys():
            Dindex = len(graph[node])
            cur_len = len(D)
            if cur_len <= Dindex:
                while cur_len <= Dindex:
                    D.append([])
                    cur_len += 1
            D[Dindex].append(node)
        
        # Add the vertex with lowest degeneracy to degen_order
        for i in range(len(D)):
            if len(D[i]) != 0:
                v = D[i].pop(0)
                degen_order += [v]
                prune_graph(graph,[v])
                
    return degen_order


def degree_ordering(G):
    nodes = list(G.keys())
    return sorted(nodes, reverse=True, key=lambda n: len(G[n]))


def BronKerbosch_pivot(G,R,P,X,cliques):
    """
    For a given graph, G, find a maximal clique containing all of the vertices
    in R, some of the vertices in P, and none of the vertices in X.
    """
    if len(P) == 0 and len(X) == 0:
        # Termination case. If P and X are empty, R is a maximal clique
        cliques.append(R)
    else:
        # choose a pivot vertex
        pivot = next(iter(P.union(X)))
        # Recurse
        for v in P.difference(G[pivot]):
            # Recursion case. 
            BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                               X.intersection(G[v]),cliques)
            P.remove(v)
            X.add(v)


def NetworkX_approximate_clique_cover(graph_dict):
    """
    NetworkX poly-time heuristic is based on
    Boppana, R., & Halldórsson, M. M. (1992).
    Approximating maximum independent sets by excluding subgraphs.
    BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    G = nx.Graph()
    for src in graph_dict:
        for dst in graph_dict[src]:
            G.add_edge(src, dst)
    return approximation.clique_removal(G)[1]


def BronKerbosch(G):
    """
    Implementation of Bron-Kerbosch algorithm (Bron, Coen; Kerbosch, Joep (1973),
    "Algorithm 457: finding all cliques of an undirected graph", Commun. ACM,
    ACM, 16 (9): 575–577, doi:10.1145/362342.362367.) using a degree ordering
    of the vertices in G instead of a degeneracy ordering.
    See: https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm
    """

    max_cliques = []

    while len(G) > 0:
        P = set(G.keys())
        R = set()
        X = set()
        v = degree_ordering(G)[0]
        cliques = []
        BronKerbosch_pivot(G,R.union({v}),P.intersection(G[v]),
                           X.intersection(G[v]),cliques)

        #print('i = {}, current v = {}'.format(i,v))
        #print('# cliques: ',len(cliques))

        sorted_cliques = sorted(cliques, key=len, reverse=True)
        max_cliques += [sorted_cliques[0]]
        #print(sorted_cliques[0])

        prune_graph(G,sorted_cliques[0])

    return max_cliques


def genMeasureCircuit(H, Nq, commutativity_type, clique_cover_method=BronKerbosch):
    """
    Take in a given Hamiltonian, H, and produce the minimum number of 
    necessary circuits to measure each term of H.

    Returns:
        List[QuantumCircuits]
    """

    start_time = time.time()

    term_reqs = np.full((len(H[1:]),Nq),'*',dtype=str)
    for i, term in enumerate(H[1:]):
        for op in term[1]:
            qubit_index = int(op[1:])
            basis = op[0]
            term_reqs[i][qubit_index] = basis

    # Generate a graph representing the commutativity of the Hamiltonian terms
    comm_graph = commutativity_type.gen_comm_graph(term_reqs)

    # Find a set of cliques within the graph where the nodes in each clique
    # are disjoint from one another.
    try:
        max_cliques = clique_cover_method(comm_graph)
    except RecursionError as re:
        print('Maximum recursion depth reached: {}'.format(re.args[0]))
        return 0, 0, 0

    end_time = time.time()

    print('MEASURECIRCUIT: {} found {} unique circuits'.format(
        clique_cover_method.__name__, len(max_cliques)))
    et = end_time - start_time
    print('MEASURECIRCUIT: Elapsed time: {:.6f}s'.format(et))
    return len(comm_graph), len(max_cliques), et


def parseHamiltonian(myPath):
    H = []
    with open(myPath) as hFile:
        for i, line in enumerate(hFile):
            line = line.split()
            if i is not 0:
                if "j" in line[0]:
                    print('Imaginary coefficient! -- skipping for now')
                    coef = 0.1
                else:
                    coef = float(line[0])
                ops = line[1:]
                H += [(coef, ops)]

    return H


def main():

    args = parse_args()

    # set the algorithm
    if args.algorithm == 'BK':
        cover_method = BronKerbosch
        cover_str = 'BronKerbosch'
    elif args.algorithm == 'BH':
        cover_method = NetworkX_approximate_clique_cover
        cover_str = 'BoppanaHalldorsson'
    else:
        print('ERROR: unrecognized MIN-CLIQUE-COVER algorithm: {}'.format(
                        args.algorithm))
        sys.exit(2)

    # set the commutation type
    if args.commutation == 'QWC':
        commutativity_type = QWCCommutativity
        type_str = 'QWC'
    elif args.commutation == 'FULL':
        commutativity_type = FullCommutativity
        type_str = 'FULL'
    else:
        print('ERROR: unrecognized commutativity type: {}'.format(
                        args.commutation))
        sys.exit(2)

    if args.specificH is None:
        # get hamiltonians
        hfiles_temp = glob.glob('hamiltonians/*')
        hfiles = [h for h in hfiles_temp if not 'taper' in h]

        # collect complexity data
        data = []
        for hfile in hfiles:
            print('--------')
            print(hfile)
            H = parseHamiltonian(hfile)

            ops = [term[1] for term in H]
            Nq = max([int(op[-1][1:]) for op in ops]) + 1
            print('{} qubits'.format(Nq))

            total_terms = len(H)
            print('{} total terms\n'.format(total_terms))
            if total_terms > 1000:
                print('Number of terms = {} > 1000'.format(total_terms))
                print('Recommend running this Hamiltonian individually')
                continue

            print(type_str + 'Commutation:')
            print(cover_str + ' algorithm:')
            num_nodes, cliques, runtime = genMeasureCircuit(
                                       H, Nq,commutativity_type,
                                       clique_cover_method=cover_method)
            print()

            data.append((num_nodes, cliques, runtime))

    else:
        hfile = args.specificH
        print('--------')
        print(hfile)
        H = parseHamiltonian(hfile)

        ops = [term[1] for term in H]
        Nq = max([int(op[-1][1:]) for op in ops]) + 1
        print('{} qubits'.format(Nq))

        total_terms = len(H)
        print('{} total terms\n'.format(total_terms))

        print(type_str + 'Commutation:')
        print(cover_str + ' algorithm:')
        num_nodes, cliques, runtime = genMeasureCircuit(
                                       H, Nq,commutativity_type,
                                       clique_cover_method=cover_method)
        print()

        data = [(num_nodes, cliques, runtime)]

    # write the results to file
    filename = 'Data/{}_{}_results.txt'.format(cover_str, type_str)
    if not(args.specificH is None):
        filename = 'Data/{}_{}_{}term_results.txt'.format(cover_str,
                                                          type_str,
                                                          total_terms)
    with open(filename, 'w') as fn:
        for run in data:
            nterms, ncliques, runtime = run
            fn.write('{0} {1} {2:.6f}\n'.format(
                                            str(nterms).ljust(5),
                                            str(ncliques).ljust(5), runtime))


if __name__ == '__main__':
    main()

