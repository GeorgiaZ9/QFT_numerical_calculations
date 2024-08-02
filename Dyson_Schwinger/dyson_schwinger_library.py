import os
import sys
import sympy as sp
from sympy import I, sqrt, conjugate

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import standard_formulas
import standard_formulas as SF


class Dyson_Schwinger_QED:
    def __init__(self):
        """
        Initialization of the symbols and formulas used from the standard_fromula library
        """
        self.SF = SF.standard_formulas()

    def ffb(self, psi1, psi2):
        """
        fermion + fermion -> boson: psi1, psi2 are u or u_bar --
        - P_b: momentum of outgoing boson --not defined yet
        """
        V = []
        for mu in range(4):
            term = (
                psi1.transpose()
                * self.SF.gamma[mu]
                * (self.SF.gR * self.SF.omega_R + self.SF.gL * self.SF.omega_L)
                * psi2
            )
            V.append(term[0])

        # Convert to light-cone representation
        V_light_cone = self.SF.four_vector_light_cone(V)

        # Simplify each component
        V_simplified = [sp.simplify(component) for component in V_light_cone]

        return sp.Matrix(V_simplified)

    def fbf(self, P_f, b, psi, antispinor=False):
        """
        (anti)fermion + boson -> (anti)fermion ---
        - P_f : the momentum of the outgoing fermion
        - antispinor (boolean) : False for spinor, True for antispinor
        """

        # Calculate P_slash (P_tot)
        P_tot = P_f[0] * self.SF.gamma[0]  # Start with the first term
        for mu in range(1, 4):
            P_tot += P_f[mu] * self.SF.gamma[mu]  # Add the rest of the terms

        b_tot = b[0] * self.SF.gamma[0]  # Start with the first term
        for mu in range(1, 4):
            b_tot += b[mu] * self.SF.gamma[mu]

        # Define the spinor u
        u = []
        if antispinor == False:
            for mu in range(4):
                term = (P_tot + self.SF.m * sp.eye(4)) * b_tot * self.SF.omega_R * psi
                u.append(term[mu])
        else:
            for mu in range(4):
                term = (
                    psi.transpose()
                    * self.SF.omega_L
                    * b_tot
                    * (P_tot + self.SF.m * sp.eye(4)).transpose()
                )
                u.append(term[mu])

        u_simplified = [sp.simplify(component) for component in u]

        return sp.Matrix(u_simplified)

    def Dyson_Schwinger(self, particle1, u1, particle2, u2, particle3, u3, particle4, u4):
        """
        u1, u2 = Bar(u2), u3, u4 = Bar(u4)
        Calculates the total amplitude using Dyson - Schwinger recursive algorithm
        """

        if incoming_particles in ("e+e-", "e-e+"):

            zp0 = self.SF.four_vector_light_cone([self.SF.p0, self.SF.px, self.SF.py, self.SF.pz])

            if outgoing_particles in ("e+e-", "e-e+"):
                b34 = self.ffb(
                    u3, u4
                )  # combo of e-e+ outgoing particles into a bosonic current b_μ
                anti_f234 = self.fbf(
                    zp0, b34, u2
                )  # combo of bosonic current from before (result from combo e+e-)

                amplitude = anti_f234 * u1

            if outgoing_particles in ("mu+mu-", "mu-mu+"):
                b34 = self.ffb(
                    u3, u4
                )  # combo of mu-mu+ outgoing particles into a bosonic current b_μ
                anti_f234 = self.fbf(
                    zp0, u2, b34
                )  # combo of bosonic current from before (result from combo e+e-)

                amplitude = anti_f234 * u1

            if outgoing_particles in ("tau+tau-", "tau-tau+"):
                b34 = self.ffb(
                    u3, u4
                )  # combo of tau-tau+ outgoing particles into a bosonic current b_μ
                anti_f234 = self.fbf(
                    zp0, u2, b34
                )  # combo of bosonic current from before (result from combo e+e-)

                amplitude = anti_f234 * u1

    # def Dyson_Schwinger(self, incoming_particles, u1, u2, outgoing_particles, u3, u4):
    #     """
    #     u1, u2 = Bar(u2), u3, u4 = Bar(u4)
    #     Calculates the total amplitude using Dyson - Schwinger recursive algorithm
    #     """

    #     if incoming_particles in ("e+e-", "e-e+"):

    #         zp0 = self.four_vector_light_cone([self.p0, self.px, self.py, self.pz])

    #         if outgoing_particles in ("e+e-", "e-e+"):
    #             b34 = self.ffb(u3, u4) #combo of e-e+ outgoing particles into a bosonic current b_μ
    #             anti_f234 = self.fbf(zp0, b34, u2) #combo of bosonic current from before (result from combo e+e-)

    #             amplitude = anti_f234 * u1

    #         if outgoing_particles in ("mu+mu-", "mu-mu+"):
    #             b34 = self.ffb(u3, u4) #combo of mu-mu+ outgoing particles into a bosonic current b_μ
    #             anti_f234 = self.fbf(zp0, u2, b34) #combo of bosonic current from before (result from combo e+e-)

    #             amplitude = anti_f234 * u1

    #         if outgoing_particles in ("tau+tau-", "tau-tau+"):
    #             b34 = self.ffb(u3, u4) #combo of tau-tau+ outgoing particles into a bosonic current b_μ
    #             anti_f234 = self.fbf(zp0, u2, b34) #combo of bosonic current from before (result from combo e+e-)

    #             amplitude = anti_f234 * u1
