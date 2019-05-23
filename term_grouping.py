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


import sys
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import pprint
import copy


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


def genMeasureCircuit(H, Nq):
    ''' Take in a given Hamiltonian, H, and produce the minimum number of 
    necessary circuits to measure each term of H.

    Returns:
        List[QuantumCircuits]
    '''

    term_reqs = np.full((len(H[1:]),Nq),'*',dtype=str)
    for i, term in enumerate(H[1:]):
        for op in term[1]:
            qubit_index = int(op[1])
            basis = op[0]
            term_reqs[i][qubit_index] = basis

    # Generate a graph representing the commutativity of the Hamiltonian terms
    comm_graph = gen_comm_graph(term_reqs)

    # Find a set of cliques within the graph where the nodes in each clique
    # are disjoint from one another.
    max_cliques = BronKerbosch(comm_graph)
    print('MEASURECIRCUIT: BronKerbosch found {} unique circuits'.format(len(max_cliques)))
    
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


if __name__ == "__main__":
  #H = [(5.076946850678632, ['I0']), (-0.006811585442824442, ['X0', 'X1', 'Y2', 'Y3']), (0.006811585442824442, ['X0', 'Y1', 'Y2', 'X3']), (0.006811585442824442, ['Y0', 'X1', 'X2', 'Y3']), (-0.006811585442824442, ['Y0', 'Y1', 'X2', 'X3']), (-0.4131939082582367, ['Z0']), (0.24086324970819822, ['Z0', 'Z1']), (0.09808757340260295, ['Z0', 'Z2']), (0.10489915884542617, ['Z0', 'Z3']), (-0.41319390825823665, ['Z1']), (0.10489915884542617, ['Z1', 'Z2']), (0.09808757340260295, ['Z1', 'Z3']), (-0.6600956717966215, ['Z2']), (0.09255914539024543, ['Z2', 'Z3']), (-0.6600956717966215, ['Z3'])]
  H = [(1, ['I0']), (1, ['X0', 'X1', 'Y2', 'Y3']), (1, ['X0', 'X1', 'Y2', 'Z3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['X0', 'X1', 'X3', 'Z4', 'Z5', 'X6']), (1, ['X0', 'X1', 'Y4', 'Y5']), (1, ['X0', 'X1', 'Y6', 'Y7']), (1, ['X0', 'Y1', 'Y2', 'X3']), (1, ['X0', 'Y1', 'Y2', 'Z3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['X0', 'Y1', 'Y3', 'Z4', 'Z5', 'X6']), (1, ['X0', 'Y1', 'Y4', 'X5']), (1, ['X0', 'Y1', 'Y6', 'X7']), (1, ['X0', 'Z1', 'X2', 'X3', 'Z4', 'X5']), (1, ['X0', 'Z1', 'X2', 'Y3', 'Z4', 'Y5']), (1, ['X0', 'Z1', 'X2', 'X4', 'Z5', 'X6']), (1, ['X0', 'Z1', 'X2', 'Y4', 'Z5', 'Y6']), (1, ['X0', 'Z1', 'X2', 'X5', 'Z6', 'X7']), (1, ['X0', 'Z1', 'X2', 'Y5', 'Z6', 'Y7']), (1, ['X0', 'Z1', 'Y2', 'Y4', 'Z5', 'X6']), (1, ['X0', 'Z1', 'Z2', 'X3', 'Y4', 'Z5', 'Z6', 'Y7']), (1, ['X0', 'Z1', 'Z2', 'X3', 'X5', 'X6']), (1, ['X0', 'Z1', 'Z2', 'Y3', 'Y4', 'Z5', 'Z6', 'X7']), (1, ['X0', 'Z1', 'Z2', 'Y3', 'Y5', 'X6']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'X4']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'X4', 'Z5']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'X4', 'Z6']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'X4', 'Z7']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'Z4', 'X5', 'Y6', 'Y7']), (1, ['X0', 'Z1', 'Z2', 'Z3', 'Z4', 'Y5', 'Y6', 'X7']), (1, ['X0', 'Z1', 'Z2', 'X4']), (1, ['X0', 'Z1', 'Z3', 'X4']), (1, ['X0', 'Z2', 'Z3', 'X4']), (1, ['Y0', 'X1', 'X2', 'Y3']), (1, ['Y0', 'X1', 'X2', 'Z3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['Y0', 'X1', 'X3', 'Z4', 'Z5', 'Y6']), (1, ['Y0', 'X1', 'X4', 'Y5']), (1, ['Y0', 'X1', 'X6', 'Y7']), (1, ['Y0', 'Y1', 'X2', 'X3']), (1, ['Y0', 'Y1', 'X2', 'Z3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['Y0', 'Y1', 'Y3', 'Z4', 'Z5', 'Y6']), (1, ['Y0', 'Y1', 'X4', 'X5']), (1, ['Y0', 'Y1', 'X6', 'X7']), (1, ['Y0', 'Z1', 'X2', 'X4', 'Z5', 'Y6']), (1, ['Y0', 'Z1', 'Y2', 'X3', 'Z4', 'X5']), (1, ['Y0', 'Z1', 'Y2', 'Y3', 'Z4', 'Y5']), (1, ['Y0', 'Z1', 'Y2', 'X4', 'Z5', 'X6']), (1, ['Y0', 'Z1', 'Y2', 'Y4', 'Z5', 'Y6']), (1, ['Y0', 'Z1', 'Y2', 'X5', 'Z6', 'X7']), (1, ['Y0', 'Z1', 'Y2', 'Y5', 'Z6', 'Y7']), (1, ['Y0', 'Z1', 'Z2', 'X3', 'X4', 'Z5', 'Z6', 'Y7']), (1, ['Y0', 'Z1', 'Z2', 'X3', 'X5', 'Y6']), (1, ['Y0', 'Z1', 'Z2', 'Y3', 'X4', 'Z5', 'Z6', 'X7']), (1, ['Y0', 'Z1', 'Z2', 'Y3', 'Y5', 'Y6']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Y4']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Y4', 'Z5']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Y4', 'Z6']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Y4', 'Z7']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Z4', 'X5', 'X6', 'Y7']), (1, ['Y0', 'Z1', 'Z2', 'Z3', 'Z4', 'Y5', 'X6', 'X7']), (1, ['Y0', 'Z1', 'Z2', 'Y4']), (1, ['Y0', 'Z1', 'Z3', 'Y4']), (1, ['Y0', 'Z2', 'Z3', 'Y4']), (1, ['Z0']), (1, ['Z0', 'X1', 'Z2', 'Z3', 'Z4', 'X5']), (1, ['Z0', 'Y1', 'Z2', 'Z3', 'Z4', 'Y5']), (1, ['Z0', 'Z1']), (1, ['Z0', 'X2', 'Z3', 'Z4', 'Z5', 'X6']), (1, ['Z0', 'Y2', 'Z3', 'Z4', 'Z5', 'Y6']), (1, ['Z0', 'Z2']), (1, ['Z0', 'X3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['Z0', 'Y3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['Z0', 'Z3']), (1, ['Z0', 'Z4']), (1, ['Z0', 'Z5']), (1, ['Z0', 'Z6']), (1, ['Z0', 'Z7']), (1, ['X1', 'X2', 'Y3', 'Y4']), (1, ['X1', 'X2', 'X4', 'Z5', 'Z6', 'X7']), (1, ['X1', 'X2', 'Y5', 'Y6']), (1, ['X1', 'Y2', 'Y3', 'X4']), (1, ['X1', 'Y2', 'Y4', 'Z5', 'Z6', 'X7']), (1, ['X1', 'Y2', 'Y5', 'X6']), (1, ['X1', 'Z2', 'X3', 'X4', 'Z5', 'X6']), (1, ['X1', 'Z2', 'X3', 'Y4', 'Z5', 'Y6']), (1, ['X1', 'Z2', 'X3', 'X5', 'Z6', 'X7']), (1, ['X1', 'Z2', 'X3', 'Y5', 'Z6', 'Y7']), (1, ['X1', 'Z2', 'Y3', 'Y5', 'Z6', 'X7']), (1, ['X1', 'Z2', 'Z3', 'X4', 'X6', 'X7']), (1, ['X1', 'Z2', 'Z3', 'Y4', 'Y6', 'X7']), (1, ['X1', 'Z2', 'Z3', 'Z4', 'X5']), (1, ['X1', 'Z2', 'Z3', 'Z4', 'X5', 'Z6']), (1, ['X1', 'Z2', 'Z3', 'Z4', 'X5', 'Z7']), (1, ['X1', 'Z2', 'Z3', 'X5']), (1, ['X1', 'Z2', 'Z4', 'X5']), (1, ['X1', 'Z3', 'Z4', 'X5']), (1, ['Y1', 'X2', 'X3', 'Y4']), (1, ['Y1', 'X2', 'X4', 'Z5', 'Z6', 'Y7']), (1, ['Y1', 'X2', 'X5', 'Y6']), (1, ['Y1', 'Y2', 'X3', 'X4']), (1, ['Y1', 'Y2', 'Y4', 'Z5', 'Z6', 'Y7']), (1, ['Y1', 'Y2', 'X5', 'X6']), (1, ['Y1', 'Z2', 'X3', 'X5', 'Z6', 'Y7']), (1, ['Y1', 'Z2', 'Y3', 'X4', 'Z5', 'X6']), (1, ['Y1', 'Z2', 'Y3', 'Y4', 'Z5', 'Y6']), (1, ['Y1', 'Z2', 'Y3', 'X5', 'Z6', 'X7']), (1, ['Y1', 'Z2', 'Y3', 'Y5', 'Z6', 'Y7']), (1, ['Y1', 'Z2', 'Z3', 'X4', 'X6', 'Y7']), (1, ['Y1', 'Z2', 'Z3', 'Y4', 'Y6', 'Y7']), (1, ['Y1', 'Z2', 'Z3', 'Z4', 'Y5']), (1, ['Y1', 'Z2', 'Z3', 'Z4', 'Y5', 'Z6']), (1, ['Y1', 'Z2', 'Z3', 'Z4', 'Y5', 'Z7']), (1, ['Y1', 'Z2', 'Z3', 'Y5']), (1, ['Y1', 'Z2', 'Z4', 'Y5']), (1, ['Y1', 'Z3', 'Z4', 'Y5']), (1, ['Z1']), (1, ['Z1', 'X2', 'Z3', 'Z4', 'Z5', 'X6']), (1, ['Z1', 'Y2', 'Z3', 'Z4', 'Z5', 'Y6']), (1, ['Z1', 'Z2']), (1, ['Z1', 'X3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['Z1', 'Y3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['Z1', 'Z3']), (1, ['Z1', 'Z4']), (1, ['Z1', 'Z5']), (1, ['Z1', 'Z6']), (1, ['Z1', 'Z7']), (1, ['X2', 'X3', 'Y4', 'Y5']), (1, ['X2', 'X3', 'Y6', 'Y7']), (1, ['X2', 'Y3', 'Y4', 'X5']), (1, ['X2', 'Y3', 'Y6', 'X7']), (1, ['X2', 'Z3', 'X4', 'X5', 'Z6', 'X7']), (1, ['X2', 'Z3', 'X4', 'Y5', 'Z6', 'Y7']), (1, ['X2', 'Z3', 'Z4', 'Z5', 'X6']), (1, ['X2', 'Z3', 'Z4', 'Z5', 'X6', 'Z7']), (1, ['X2', 'Z3', 'Z4', 'X6']), (1, ['X2', 'Z3', 'Z5', 'X6']), (1, ['X2', 'Z4', 'Z5', 'X6']), (1, ['Y2', 'X3', 'X4', 'Y5']), (1, ['Y2', 'X3', 'X6', 'Y7']), (1, ['Y2', 'Y3', 'X4', 'X5']), (1, ['Y2', 'Y3', 'X6', 'X7']), (1, ['Y2', 'Z3', 'Y4', 'X5', 'Z6', 'X7']), (1, ['Y2', 'Z3', 'Y4', 'Y5', 'Z6', 'Y7']), (1, ['Y2', 'Z3', 'Z4', 'Z5', 'Y6']), (1, ['Y2', 'Z3', 'Z4', 'Z5', 'Y6', 'Z7']), (1, ['Y2', 'Z3', 'Z4', 'Y6']), (1, ['Y2', 'Z3', 'Z5', 'Y6']), (1, ['Y2', 'Z4', 'Z5', 'Y6']), (1, ['Z2']), (1, ['Z2', 'X3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['Z2', 'Y3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['Z2', 'Z3']), (1, ['Z2', 'Z4']), (1, ['Z2', 'Z5']), (1, ['Z2', 'Z6']), (1, ['Z2', 'Z7']), (1, ['X3', 'X4', 'Y5', 'Y6']), (1, ['X3', 'Y4', 'Y5', 'X6']), (1, ['X3', 'Z4', 'Z5', 'Z6', 'X7']), (1, ['X3', 'Z4', 'Z5', 'X7']), (1, ['X3', 'Z4', 'Z6', 'X7']), (1, ['X3', 'Z5', 'Z6', 'X7']), (1, ['Y3', 'X4', 'X5', 'Y6']), (1, ['Y3', 'Y4', 'X5', 'X6']), (1, ['Y3', 'Z4', 'Z5', 'Z6', 'Y7']), (1, ['Y3', 'Z4', 'Z5', 'Y7']), (1, ['Y3', 'Z4', 'Z6', 'Y7']), (1, ['Y3', 'Z5', 'Z6', 'Y7']), (1, ['Z3']), (1, ['Z3', 'Z4']), (1, ['Z3', 'Z5']), (1, ['Z3', 'Z6']), (1, ['Z3', 'Z7']), (1, ['X4', 'X5', 'Y6', 'Y7']), (1, ['X4', 'Y5', 'Y6', 'X7']), (1, ['Y4', 'X5', 'X6', 'Y7']), (1, ['Y4', 'Y5', 'X6', 'X7']), (1, ['Z4']), (1, ['Z4', 'Z5']), (1, ['Z4', 'Z6']), (1, ['Z4', 'Z7']), (1, ['Z5']), (1, ['Z5', 'Z6']), (1, ['Z5', 'Z7']), (1, ['Z6']), (1, ['Z6', 'Z7']), (1, ['Z7'])]
  Nq = 8
  circlist = genMeasureCircuit(H, Nq)
  names = [c[1] for c in circlist]
  print(names)











