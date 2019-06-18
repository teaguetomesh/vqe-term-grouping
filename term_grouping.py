'''
Teague Tomesh - 4/26/2019

Given a particular qubit Hamiltonian, measuring the expected energy of any
given quantum state will depend only on the individual terms of that 
Hamiltonian. 

measureCircuit.py generates a circuit which will measure a quantum state in the
correct bases to allow the energy to be calculated. This may require generating
multiple circuits if the same qubit needs to be measured in two perpendicular
bases (i.e. Z and X).

To find the minimum number of circuits needed to measure an entire Hamiltonian,
we treat the terms of H as nodes in a graph, G, where there are edges between
nodes indicate those two terms commute with one another. Finding the circuits
now becomes a clique finding problem which can be solved by the 
BronKerbosch algorithm.

'''


import time
import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import pprint
import copy


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
    '''
    Produce a degeneracy ordering of the vertices in graph, as outlined in,
    Eppstein et. al. (arXiv:1006.5440)
    '''

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
    '''
    For a given graph, G, find a maximal clique containing all of the vertices
    in R, some of the vertices in P, and none of the vertices in X.
    '''
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


def BronKerbosch(G):
    '''
    Implementation of Bron-Kerbosch algorithm using a degree ordering of the
    vertices in G instead of a degeneracy ordering.
    See: https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm
    '''
    
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


def generate_circuit_matrix(Nq, max_cliques):
    circuitMatrix = np.empty((Nq,len(max_cliques)),dtype=str)
    for i, clique in enumerate(max_cliques):
        # each clique will get its own circuit
        clique_list = list(clique)

        # Take the first string to be the circuit template, i.e. '****Z**Z'
        circStr = list(clique_list[0])
        #print(circStr)

        # Loop through the characters of the template and replace the '*'s
        # with a X,Y,Z found in another string in the same clique
        for j, char in enumerate(circStr):
            if char == '*':
                #print('j = {}, c = {}'.format(j,char))
                for tstr in clique_list[1:]:
                    # Search through the remaining strings in the clique
                    #print('tstr = {}, {} != * = {}'.format(tstr, tstr[j], (tstr[j] != '*')))
                    if tstr[j] != '*':
                        circStr[j] = tstr[j]
                        break
                
                if circStr[j] == '*':
                    # After searching through all of the strings in the clique
                    # the current char is still '*', this means none of the
                    # terms in this clique depend on this qubit -> measure in
                    # the Z basis.
                    circStr[j] = 'Z'

        # Once the circuit string is filled in, add it to the circuit matrix
        for q in range(Nq):
            circuitMatrix[q,i] = circStr[q]

    return circuitMatrix


def genMeasureCircuit(H, Nq, commutativity_type):
    ''' Take in a given Hamiltonian, H, and produce the minimum number of 
    necessary circuits to measure each term of H.

    Returns:
        List[QuantumCircuits]
    '''

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
    max_cliques = BronKerbosch(comm_graph)

    end_time = time.time()

    print('MEASURECIRCUIT: BronKerbosch found {} unique circuits'.format(len(max_cliques)))
    et = end_time - start_time
    print('MEASURECIRCUIT: Elapsed time: {:.6f}s'.format(et))
    return max_cliques

    '''
    This section of the code produces a quantum circuit to measure each of
    the cliques found above. However, it can only handle QWC cliques since
    commuting groups like [XX,YY,ZZ] require a bit more handling.

    # Generate the circuitMatrix
    #   Number of rows = number of qubits
    #   Number of cols = number of circuits
    #   Each entry will have a value v <- {'Z','X','Y'} corresponding to the
    #   basis this qubit must be measured in.
    circuitMatrix = generate_circuit_matrix(Nq, max_cliques)

    # After the circuit matrix is computed, construct the quantum circuits
    circuitList = []
    for m in range(circuitMatrix.shape[1]):
        qr = QuantumRegister(Nq, name='qreg')
        cr = ClassicalRegister(Nq, name='creg')
        circ = QuantumCircuit(qr, cr)
        name = ''
        circ.barrier(qr)
        for n, basis in enumerate(circuitMatrix[:,m]):
            if basis == 'X':
                circ.h(qr[n])
            elif basis == 'Y':
                circ.sdg(qr[n])
                circ.h(qr[n])
            if basis == '*': basis = 'Z'
            name += basis
        circ.barrier(qr)
        for n in range(Nq):
            circ.measure(qr[n],cr[n])
        #print('name: ',name)
        circuitList += [(circ, name)]
        #circ.draw(scale=0.8, filename='throwaway/measure_{}_single_circ{}'.format(name,m), 
        #    output='mpl', plot_barriers=False, reverse_bits=True)

    return circuitList
    '''


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


if __name__ == "__main__":
    # change the number of qubits based on which hamiltonian is selected
    hfile = 'hamiltonians/sampleH2.txt'
    H = parseHamiltonian(hfile)

    # Infer number of qubits from widest term in Hamiltonian
    ops = [term[1] for term in H]
    Nq = max([len(op) for op in ops])
    print('%s qubits' % Nq)

    for commutativity_type in [QWCCommutativity, FullCommutativity]:
        cliques = genMeasureCircuit(H, Nq, commutativity_type)
        for cliq in cliques:
            print(cliq)
        print()
