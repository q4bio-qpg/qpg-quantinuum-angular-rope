import math
import numpy as np
import sympy as sp
from pytket.circuit import Circuit


# ------------------ Base components ------------------

def _ry_rz_ry(circ: Circuit, theta, phi, psi, q):
    circ.Ry(theta, q)
    circ.Rz(phi,   q)
    circ.Ry(psi,   q)


def euler_cartan_block(circ: Circuit, a, b, params):
    """Euler–Cartan block on qubits (a,b), 15 parameters total."""
    (θa1, φa1, ψa1,
     θb1, φb1, ψb1,
     α, β, γ,
     θa2, φa2, ψa2,
     θb2, φb2, ψb2) = params

    _ry_rz_ry(circ, θa1, φa1, ψa1, a)
    _ry_rz_ry(circ, θb1, φb1, ψb1, b)
    circ.TK2(α, β, γ, a, b)
    _ry_rz_ry(circ, θa2, φa2, ψa2, a)
    _ry_rz_ry(circ, θb2, φb2, ψb2, b)


def euler_cartan_layer_add(circ: Circuit, symbols, s=0, add_hadamar=True):
    """Add one Euler–Cartan layer; shift=0 for (0,1),(2,3)..., shift=1 for (1,2),(3,4)..."""
    q = circ.n_qubits
    n_pairs = q // 2
    assert len(symbols) == 15 * n_pairs, f"Expected {15*n_pairs} params for {n_pairs} pairs"
    idx = 0
    for j in range(s, s + q, 2):
        a = j % q
        b = (j + 1) % q
        block_syms = symbols[idx:idx+15]
        euler_cartan_block(circ, a, b, block_syms)
        idx += 15

    if add_hadamar:
        for qid in range(q):
            circ.H(qid)


# ------------------ Main builder ------------------

def euler_cartan_circuit(q: int, n: int, add_hadamar=True):
    """
    Build a pytket Circuit with q qubits and n (effective) parameters,
    using Euler–Cartan layers. Pads unused parameters with zeros.

    Args:
        q (int): number of qubits (must be even)
        n (int): number of parameters actually used

    Returns:
        (Circuit, List[sympy.Symbol]): circuit and full symbol list
    """
    assert q % 2 == 0, "Number of qubits must be even."

    params_per_layer = 15 * (q // 2)
    n_layers = math.ceil(n / params_per_layer)

    # full symbol list for all layers
    total_params = params_per_layer * n_layers
    symbols = [sp.symbols(f"t{i}") for i in range(total_params)]

    circ = Circuit(q)
    for k in range(n_layers):
        layer_syms = symbols[k*params_per_layer:(k+1)*params_per_layer]
        euler_cartan_layer_add(circ, layer_syms, s=(k % 2), add_hadamar=add_hadamar)

    # zero-pad tail (unused parameters)
    tail_zeros = {symbols[i]: 0 for i in range(n, total_params)}
    circ.symbol_substitution(tail_zeros)

    return circ, symbols


# ------------------ Encoder class ------------------

class EulerCartanEncoder:
    """
    Universal Euler–Cartan encoder using pytket.
    Parameters are in half-turns (units of π); multiply by 1/π if your data are radians.
    """

    def __init__(self, n: int, q: int, scale: float = 1.0, data_in_radians=False, add_hadamar=True):
        """
        Args:
            n (int): number of input parameters
            q (int): number of qubits (even)
            scale (float): scaling factor applied to parameters before substitution
        """
        self.n = int(n)
        self.q = int(q)
        if data_in_radians:
            scale = scale/np.pi
        self.scale = float(scale)
        self.circuit, self.symbols = euler_cartan_circuit(self.q, self.n, add_hadamar=add_hadamar)

    def __call__(self, params):
        assert len(params) == self.n, "Number of parameters mismatch."
        params = np.asarray(params) * self.scale
        sym_map = {self.symbols[i]: params[i] for i in range(self.n)}
        self.circuit.symbol_substitution(sym_map)

    def dispatch(self, params):
        """Return a new pytket circuit with parameters applied."""
        assert len(params) == self.n, "Number of parameters mismatch."
        params = np.asarray(params) * self.scale
        sym_map = {self.symbols[i]: params[i] for i in range(self.n)}
        circ_new = self.circuit.copy()
        circ_new.symbol_substitution(sym_map)
        return circ_new


################ ANGULAR Encoder ################

def angular2_layer_add(circuit, symbols, s=0, skip=0):
    q = circuit.n_qubits
    qubits = circuit.qubits
    assert q % 2 == 0, "Number of qubits must be even"

    if s==0:
        for j in range(int(q/2)-skip):
            circuit.ZZMax(qubits[2*j], qubits[2*j+1])
    else: 
        for j in range(int(q/2)-skip):
            circuit.ZZMax(qubits[2*j+1], qubits[(2*j+2)%q])
        
    for j in range(q-2*skip):
        circuit.H(qubits[j])
        circuit.Ry(symbols[2*j], qubits[j])        
        circuit.Rx(symbols[2*j+1], qubits[j])   

def angular2_circuit(q, n):   
    assert n % 4 == 0, "Number of params isn't divided by 4"
    n_layer = math.floor(n/q/2)
    ntail = n-2*q*n_layer  
    symbols = [sp.symbols(f'x{i}') for i in range(n)]
    circuit = Circuit(q)
    qubits = circuit.qubits

    # TODO:  another symbol order could be better
    
    # init 1-qubit layer 
    for j in range(0, q):
        circuit.Ry(symbols[2*j], qubits[j])
        circuit.Rx(symbols[2*j+1], qubits[j])

    # add layers
    for k in range(n_layer-1):     
        angular2_layer_add(circuit, symbols[2*q*(k+1):2*q*(k+2)], k%2)

    # add last 1-qubit layer 
    for j in range(int(ntail/2)):
        circuit.Ry(symbols[2*q*n_layer + 2*j], qubits[j])
        circuit.Rx(symbols[2*q*n_layer + 2*j+1], qubits[j])        

    # tail_zeros = {symbols[i]:0 for i in range(n, 2*q*n_layer)} 
    # circuit.symbol_substitution(tail_zeros)
    return (circuit, symbols)
    
class Angular2Encoder:
    """
    Angular Encoder using pytket. Supports parameter substitution and statevector simulation.
    """

    def __init__(self, n: int, q: int, scale: float):
        """
        Args:
            n (int): number of input parameters
            q (int): number of qubits
            scale (float): scaling factor
        """
        self.n = int(n)
        self.q = int(q)
        self.scale = float(scale)
        self.circuit, self.symbols = angular2_circuit(self.q, self.n)

    def __call__(self, params):
        assert len(params) == self.n, "Number of parameters mismatch"
        params = np.asarray(params) * self.scale
        sym_map = {self.symbols[i]:params[i] for i in range(len(params))}
        self.circuit.symbol_substitution(sym_map)

    def dispatch(self, params):
        """
        Return new pytket circuit with parameters applied
        """
        assert len(params) == self.n, "Number of parameters mismatch"
        params = np.asarray(params) * self.scale
        sym_map = {self.symbols[i]:params[i] for i in range(len(params))}

        circuit_new = self.circuit.copy()
        circuit_new.symbol_substitution(sym_map)

        return circuit_new
