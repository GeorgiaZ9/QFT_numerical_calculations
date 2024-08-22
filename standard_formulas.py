import sympy as sp
from sympy import I, sqrt, conjugate
import numpy as np


class standard_formulas:

    def __init__(self):

        self.setup_symbols()
        self.setup_gamma_matrices()
        self.setup_omega()

    def allow_interactions(particle1, particle2, particle3=None):
        """
        Determine if an interaction is allowed based on QED Feynman rules.

        Args:
        particle1, particle2, particle3 (str): Particle types ('e-', 'e+', 'photon')

        Returns:
        bool: True if the interaction is allowed, False otherwise
        """

        # Two-particle interactions (for initial/final states)
        if particle3 is None:
            # e- and e+ can interact
            if set([particle1, particle2]) == set(["e-", "e+"]):
                return True
            # photon can't interact with itself
            elif particle1 == "photon" and particle2 == "photon":
                return False
            # single particle can't interact with itself
            elif particle1 == particle2:
                return False
            else:
                return False

        # Three-particle vertices
        else:
            # Check if we have exactly one photon and two fermions (e- or e+)
            particles = [particle1, particle2, particle3]
            if particles.count("photon") == 1 and (
                particles.count("e-") + particles.count("e+") == 2
            ):
                return True
            else:
                return False

        return False  # Default case, shouldn't be reached

    def setup_symbols(self):
        """
        Set up the symbolic part used:
        - P = (p0, px, py, pz) momentum
        - m = mass of the particle
        - gR, gL = e for a QED (Fermion-Fermion-photon) vertex (coupling)
        - v = 246 GeV

        """
        self.px, self.py, self.pz, self.p0 = sp.symbols("px py pz p0", real=True)
        self.m = sp.Symbol("m", positive=True)
        self.gR, self.gL = sp.symbols("g_R g_L")

        self.v = sp.Symbol("v", positive=True)
        self.C_uf = sp.Symbol("C^{u\phi}")
        self.C_uG = sp.Symbol("C^{uG}")
        self.C_uG_star = sp.Symbol("C^{uG*}")
        self.C_HG = sp.Symbol("C^{\phi G}")

    def setup_gamma_matrices(self):
        """
        Setup of the Pauli σ-matrices and γ-matrices based on Weyl (chiral) basis
        (alternate form).
        """
        self.sigma1 = sp.Matrix([[0, 1], [1, 0]])
        self.sigma2 = sp.Matrix([[0, -I], [I, 0]])
        self.sigma3 = sp.Matrix([[1, 0], [0, -1]])

        self.gamma5 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        self.gamma = [
            sp.Matrix([[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]]),
            sp.Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]),
            sp.Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, I, 0, 0], [-I, 0, 0, 0]]),
            sp.Matrix([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]),
        ]

    def sigma(self, m, n):

        return I / 2 * (self.gamma[m] * self.gamma[n] - self.gamma[n] * self.gamma[m])

    def metric(self):

        return sp.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

    def setup_omega(self):
        """
        Setup of the ω_R and ω_L matrices,
        used for the left-handed and right-handed spin projections.
        """
        self.omega_R = (sp.eye(4) + self.gamma5) * 1 / 2
        self.omega_L = (sp.eye(4) - self.gamma5) * 1 / 2

    def four_vector_light_cone(self, V):
        """
        Returns the 4-vector lightcone representation vector of an input vector V.
        """
        V0, Vx, Vy, Vz = V
        return sp.Matrix([V0 + Vz, V0 - Vz, Vx + I * Vy, Vx - I * Vy])

    def inverse_lightcone_rep(self, V):
        """
        Returns the standard 4-vector representation from a light cone representation vector V.
        """
        V0, V1, V2, V3 = V

        Vz = (V0 - V1) / 2
        V0_standard = (V0 + V1) / 2
        Vx = (V2 + V3) / 2
        Vy = (V2 - V3) / (2 * sp.I)

        return sp.Matrix([V0_standard, Vx, Vy, Vz])

    def p_abs(self):
        """
        Returns the norm of a 3-vector.
        """
        return sqrt(self.px**2 + self.py**2 + self.pz**2)

    def slash(self, P_f):
        # Calculate P_slash
        P_tot = P_f[0] * self.gamma[0]
        for mu in range(1, 4):
            P_tot -= P_f[mu] * self.gamma[mu]

        return P_tot

    def polarization_vectors(self):
        """
        Returns the polarization vectors.
        """
        pT = sqrt(self.px**2 + self.py**2)
        p_abs = self.p_abs()

        eps_minus = sp.Matrix(
            [
                -pT / (sqrt(2) * p_abs),
                pT / (sqrt(2) * p_abs),
                (self.px + I * self.py) * (p_abs + self.pz) / (sqrt(2) * p_abs * pT),
                (self.px - I * self.py) * (-p_abs + self.pz) / (sqrt(2) * p_abs * pT),
            ]
        )

        eps_plus = sp.Matrix(
            [
                pT / (sqrt(2) * p_abs),
                -pT / (sqrt(2) * p_abs),
                (self.px + I * self.py) * (p_abs - self.pz) / (sqrt(2) * p_abs * pT),
                (self.px - I * self.py) * (-p_abs - self.pz) / (sqrt(2) * p_abs * pT),
            ]
        )

        eps_zero = sp.Matrix(
            [
                p_abs / (sqrt(2) * self.p0) + self.pz * self.p0 / (sqrt(2) * p_abs * self.p0),
                p_abs / (sqrt(2) * self.p0) - self.pz * self.p0 / (sqrt(2) * p_abs * self.p0),
                (self.px + I * self.py) * self.p0 / (sqrt(2) * p_abs * self.p0),
                (self.px - I * self.py) * self.p0 / (sqrt(2) * p_abs * self.p0),
            ]
        )

        return eps_minus, eps_plus, eps_zero

    def spinor_u(self, helicity):
        """
        Returns a spinor in a 4-vector form for positive or negative helicity.
        """
        a = self.p0 + self.p_abs()
        b = self.pz + self.p_abs()
        c = 2 * self.p_abs()
        r = sqrt(a * b * c)

        if helicity == "+":
            return sp.Matrix(
                [
                    r / c,
                    a * (self.px + I * self.py) / r,
                    -self.m * b / r,
                    -self.m * (self.px + I * self.py) / r,
                ]
            )
        elif helicity == "-":
            return sp.Matrix(
                [
                    self.m * (self.px - I * self.py) / r,
                    -self.m * b / r,
                    -a * (self.px - I * self.py) / r,
                    r / c,
                ]
            )

    def spinor_ubar(self, helicity):
        """
        Returns an antispinor in a 4-vector form for positive or negative helicity.
        """
        a = self.p0 + self.p_abs()
        b = self.pz + self.p_abs()
        c = 2 * self.p_abs()
        r = sqrt(a * b * c)

        if helicity == "+":
            return sp.Matrix(
                [
                    self.m * b / r,
                    self.m * (self.px - I * self.py) / r,
                    -r / c,
                    -a * (self.px - I * self.py) / r,
                ]
            ).T
        elif helicity == "-":
            return sp.Matrix(
                [
                    a * (self.px + I * self.py) / r,
                    -r / c,
                    -self.m * (self.px + I * self.py) / r,
                    self.m * b / r,
                ]
            ).T
