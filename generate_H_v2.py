from openfermion.hamiltonians import MolecularData


name = 'H2'
basis = 'sto-3g'
multiplicity = 1
num_electrons = 2

bond_length = 0.7
description = str(bond_length)
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]

myfile = 'molecule_data/{}_{}_{}_{}.hdf5'.format(name,basis,'singlet',bond_length)

molecule = MolecularData(geometry, basis, multiplicity, description=description,
                         filename=filename)
