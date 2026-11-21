"""
The quantum strategy of the six-node counterexample to moderate and strong face splitting, checked numerically.

Structure (unsplit): relay R, Alice A, Dan D, Eve E, settings S and T; edges S -> A, S -> D, T -> E, R -> D, A -> D;
sources Q{R, A, D} and Q{R, D, E}. The first source holds a Bell pair between Alice's first qubit and the relay and
a Bell pair between Alice's second qubit and Dan; the second source holds a Bell pair between the relay and Eve.

* The relay measures its two qubits in the Bell basis and outputs r (four values).
* Alice, for s in {0, 1}, measures (Z + X)/sqrt2 or (Z - X)/sqrt2 on her first qubit; for s = 2 she measures her
  two qubits in the Bell basis and outputs a (four values).
* Eve measures Z (t = 0) or X (t = 1).
* Dan, at s = 2, measures Z and multiplies by the sign he computes from r and a. (For s in {0, 1} his outcome is a
  constant; that part of the distribution plays no role and is not built here.)

Printed and saved: the distribution of r; the four correlators c_r(s, t) = <A_s E_t> given r and the CHSH value
they give with the signs of that r; at s = 2, the probability of each (r, a), the product <Z_D Z_E> and the mean
<Z_E> given (r, a). The document uses exactly these facts, its properties (P1) and (P2): r is uniform, CHSH is
2 sqrt2 for every r with four different sign patterns, and at s = 2, t = 0 Dan's corrected outcome equals Eve's,
which is uniform given (r, a).
"""
import itertools

import numpy as np

from _repo import save

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=float)
Z = np.diag([1.0, -1.0])
PAULIS = [I2, X, Z, X @ Z]                      # the Bell state k is (1 x PAULIS[k]) |Phi+>
PHI = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
A1, RF, A2, D1, RG, E = range(6)                # qubit order of the global state
N = 6


def embed(op: np.ndarray, positions) -> np.ndarray:
    """`op` acting on the qubits `positions` (in that order), identity elsewhere."""
    k = len(positions)
    rest = [i for i in range(N) if i not in positions]
    big = np.kron(op, np.eye(2 ** (N - k))).reshape([2] * (2 * N))
    inverse = np.argsort(list(positions) + rest)
    return big.transpose(list(inverse) + [N + i for i in inverse]).reshape(2 ** N, 2 ** N)


def bell_projector(k: int) -> np.ndarray:
    v = np.kron(I2, PAULIS[k]) @ PHI
    return np.outer(v, v.conj())


def main() -> dict:
    psi = np.kron(np.kron(PHI, PHI), PHI)        # (A1 RF) (A2 D1) (RG E)
    rho = np.outer(psi, psi.conj())
    relay = [embed(bell_projector(r), [RF, RG]) for r in range(4)]
    alice = [embed((Z + X) / np.sqrt(2), [A1]), embed((Z - X) / np.sqrt(2), [A1])]
    eve = [embed(Z, [E]), embed(X, [E])]
    swap = [embed(bell_projector(a), [A1, A2]) for a in range(4)]
    z_dan, z_eve = embed(Z, [D1]), eve[0]

    result = {'p_r': [], 'correlators': {}, 'chsh': {}, 'signs': {}, 's=2': {}}
    for r in range(4):
        p = float(np.trace(relay[r] @ rho).real)
        result['p_r'].append(round(p, 12))
        state = relay[r] @ rho @ relay[r] / p
        corr = [[float(np.trace(alice[s] @ eve[t] @ state).real) for t in (0, 1)] for s in (0, 1)]
        eps = (int(np.sign(corr[0][0])), int(np.sign(corr[0][1])))            # signs of c_r(0, 0), c_r(0, 1)
        chsh = eps[0] * (corr[0][0] + corr[1][0]) + eps[1] * (corr[0][1] - corr[1][1])
        result['correlators'][r] = [[round(c, 12) for c in row] for row in corr]
        result['signs'][r] = eps
        result['chsh'][r] = round(chsh, 12)
        for a in range(4):
            post = swap[a] @ relay[r] @ rho @ relay[r] @ swap[a]
            q = float(np.trace(post).real)
            post = post / q
            result['s=2'][f'{r},{a}'] = {'probability': round(q, 12),
                                         'ZZ': round(float(np.trace(z_dan @ z_eve @ post).real), 12),
                                         'Z_E': round(float(np.trace(z_eve @ post).real), 12)}

    assert all(abs(p - 0.25) < 1e-12 for p in result['p_r'])
    assert all(abs(v - 2 * np.sqrt(2)) < 1e-9 for v in result['chsh'].values())
    assert len({tuple(e) for e in result['signs'].values()}) == 4            # four distinct sign patterns
    assert all(abs(abs(v['ZZ']) - 1) < 1e-9 and abs(v['Z_E']) < 1e-9 and abs(v['probability'] - 1 / 16) < 1e-12
               for v in result['s=2'].values())
    return result


if __name__ == '__main__':
    out = main()
    print('P(r):', out['p_r'])
    for r in range(4):
        print(f"r={r}: c_r(s,t) = {out['correlators'][r]}, signs {out['signs'][r]}, CHSH = {out['chsh'][r]:.6f}")
    print('s = 2: (r, a) -> probability, <Z_D Z_E>, <Z_E>')
    for key, value in out['s=2'].items():
        print(f"   ({key}): {value['probability']:.4f}, {value['ZZ']:+.0f}, {value['Z_E']:+.0f}")
    print('saved', save('six_node_strategy.json', out))
