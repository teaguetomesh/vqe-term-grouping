'''
Teague Tomesh - 3/13/2019

Use the OpenFermion package to generate qubit Hamiltonians for a wide variety
of different molecules, geometries, and fermion-qubit mappings. 

'''


import sys
from pathlib import Path
from openfermion.hamiltonians import MolecularData
import openfermion.hamiltonians as oh
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermionpsi4 import run_psi4


def generate_and_save(geometry, basis, multiplicity, description, filename):
    
  # Initialize the molecule
  molecule = MolecularData(geometry, basis, multiplicity, description=description,
                          filename=filename)

  # Compute the active space integrals
  print('-computing integrals')
  molecule = run_psi4(molecule,run_mp2=True,run_cisd=True,run_ccsd=True,run_fci=True)
  molecule.save()
  print('Successful generation')
    

def load_and_transform(filename, orbitals, transform):
  # Load data
  print('--- Loading molecule ---')
  molecule = MolecularData(filename=filename)
  molecule.load()

  print('filename: {}'.format(molecule.filename))
  print('n_atoms: {}'.format(molecule.n_atoms))
  print('n_electrons: {}'.format(molecule.n_electrons))
  print('n_orbitals: {}'.format(molecule.n_orbitals))
  #print('Canonical Orbitals: {}'.format(molecule.canonical_orbitals))
  print('n_qubits: {}'.format(molecule.n_qubits))
  
  # Get the Hamiltonian in an active space.
  # Set Hamiltonian parameters.
  occupied_orbitals, active_orbitals = orbitals

  molecular_hamiltonian = molecule.get_molecular_hamiltonian(
		  occupied_indices=range(occupied_orbitals),
		  active_indices=range(active_orbitals))

  # Map operator to fermions and qubits.
  fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
  #print('Fermionic Hamiltonian is:\n{}'.format(fermion_hamiltonian))

  if transform is 'JW':
    qubit_h = jordan_wigner(fermion_hamiltonian)
    qubit_h.compress()
    print('\nJordan-Wigner Hamiltonian:\n{}'.format(qubit_h))
  elif transform is 'BK':
    qubit_h = bravyi_kitaev(fermion_hamiltonian)
    qubit_h.compress()
    print('\nBravyi-Kitaev Hamiltonian is:\n{}'.format(qubit_h))
  else:
    print('ERROR: Unrecognized qubit transformation: {}'.format(transform))
    sys.exit(2)

  return qubit_h


def write_to_file(filename, name, Ne, hamiltonian, description):
  
  # Write the resulting qubit H to file
  print('\n\n~~write Qubit Hamiltonian to file~~\n')
  print(filename)
  with open(filename, 'w') as H_file:
    H_file.write('{} {}\n'.format(name, Ne))
    #for h in my_Hs:
    hstring = '{}'.format(hamiltonian)
    print(hstring)
    print('')
    terms = hstring.split('\n')
    for t in terms:
      t2 = t.split('[')
      if len(t2) is 2:
        coef = t2[0]
        paul = t2[1].split(']')[0]
        # Check for identity operator
        if paul is '':
          paul = 'I0'
        
        # Write coefficients and operators to file
        H_file.write('{0:17s} {1}\n'.format(coef, paul))

      else:
        print('ERROR: Something went wrong parsing string')
  print('Successful write\n')


def main(argv):
  '''
  '''

  # Set molecule parameters.
  name = 'H2'
  basis = '6-31g'
  multiplicity = 1
  n_points = 30
  bond_length_interval = 0.1
  num_electrons = 2
  transform = 'JW'

  for occupied_num in range(5):
    for active_num in range(1,5):

      # Generate molecule at different bond lengths.
      for point in range(1, n_points + 1):
        bond_length = bond_length_interval * point
        description = str(round(bond_length,2))
        
        #geometry = [('O',(0.,0.,0.)),('H',(0.757,0.586,0.)),('H',(-0.757,0.586,0.))]
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
        
        # If this molecule has not been generated
        if multiplicity is 1:
          mult = 'singlet'
        elif multiplicity is 3:
          mult = 'triplet'
        
        molecule_file = 'molecule_data/{}_{}_{}_{}.hdf5'.format(name,basis,mult,bond_length)
        config = Path(molecule_file)
        
        if True:#not config.is_file():
          # Generate it now
          print('--- Generate Molecule: {}_{}_{:.2f} ---'.format(name,basis,bond_length))
          generate_and_save(geometry, basis, multiplicity, description, molecule_file)
        
        # Load the molecule and perform qubit transformation
        occupied = occupied_num
        active = active_num
        orbitals = (occupied, active)
        qubit_h = load_and_transform(molecule_file, orbitals, transform)

        # Write the qubit hamiltonian to file
        folder = '{}_{}_{}_OS{}/AS{}/'.format(name, basis, transform, occupied, active)
        fn = 'qubitH_{}_{}_{}_{}.txt'.format(name, basis, transform, description)
        qubit_file = folder + fn
        write_to_file(qubit_file, name, num_electrons, qubit_h, description)


if __name__ == "__main__":
  main(sys.argv[1:])





















