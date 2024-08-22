import os
import sys
import sympy as sp
from sympy import I, sqrt, conjugate
import numpy as np


# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import standard_formulas
import standard_formulas as SF


class SMEFT:
    def __init__(self):
        """
        Initialization of the symbols and formulas used from the standard_fromula library
        """
        self.SF = SF.standard_formulas()

    def uH_operator(self):
        """
        Returns the result of the uH (spinor - Higgs) coupling matrix for SMEFT.
        """

        return (
            I
            * self.SF.C_uf
            * self.SF.v**2
            / sp.sqrt(2)
            * (self.SF.omega_L * self.SF.gL + self.SF.omega_R * self.SF.gR)
        )

    def uG_operator_ff(self, p3, psi1, psi2, representation="Lorentz"):
        """
        Returns the result of the uG operator (SMEFT) applied to two incoming spinors (fermions).
        """
        if representation == "Lorentz":
            P = p3.transpose() * self.SF.metric()

        elif representation == "lightcone":
            P = self.SF.inverse_lightcone_rep(p3).transpose() * self.SF.metric()

        res = []

        for m in range(4):
            sum = sp.Matrix.zeros(4, 4)
            for n in range(4):
                term = (
                    # -sp.sqrt(2)
                    # * self.SF.v
                    # * self.SF.C_uG
                    P[n]
                    * self.SF.sigma(m, n)
                    * (self.SF.omega_L * self.SF.gL + self.SF.omega_R * self.SF.gR)
                )
                sum += term

            res.append(psi1.transpose() * sum * psi2)

        res_light = self.SF.four_vector_light_cone(res)

        return sp.Matrix(res_light)

    def uG_operator_vf(
        self, p3, v, psi2, antispinor=False, representation="Lorentz", propagator=False
    ):
        """
        Returns the result of the uG operator (SMEFT) applied to an incoming spinor (fermion)
        and an incoming vector (boson).
        """
        if representation == "Lorentz":
            P = p3.transpose() * self.SF.metric()
            g = v.transpose() * self.SF.metric()

        elif representation == "lightcone":
            P = self.SF.inverse_lightcone_rep(p3).transpose() * self.SF.metric()
            g = self.SF.inverse_lightcone_rep(v).transpose() * self.SF.metric()

        if antispinor == False:

            res = []
            sum1 = sp.Matrix.zeros(4, 4)
            for m in range(4):
                sum = sp.Matrix.zeros(4, 4)
                for n in range(4):
                    term = (
                        # -sp.sqrt(2)
                        # * self.SF.v
                        # * self.SF.C_uG
                        g[m]
                        * P[n]
                        * self.SF.sigma(m, n)
                        * (self.SF.omega_L * self.SF.gL + self.SF.omega_R * self.SF.gR)
                    )
                    sum += term

                sum1 += sum

            if propagator == False:
                res.append(sum1 * psi2)

            elif propagator == True:
                P_tot = self.SF.slash(P * self.SF.metric())

                res.append((P_tot + self.SF.m * sp.eye(4)) * sum1 * psi2)

            final_result = sp.Matrix(res)

        elif antispinor == True:
            res = []
            sum1 = sp.Matrix.zeros(4, 4)
            for m in range(4):
                sum = sp.Matrix.zeros(4, 4)
                for n in range(4):
                    term = (
                        # -sp.sqrt(2)
                        # * self.SF.v
                        # * self.SF.C_uG
                        P[n]
                        * self.SF.sigma(m, n)
                        * (self.SF.omega_L * self.SF.gL + self.SF.omega_R * self.SF.gR)
                        * g[m]
                    )
                    sum += term

                sum1 += sum

            if propagator == False:
                res.append(psi2.transpose() * sum1)

            # elif propagator == True:
            #     P_tot = self.SF.slash(P * self.SF.metric())

            #     res.append( psi2.transpose() * sum1 * (P_tot - self.SF.m * sp.eye(4)))

            final_result = sp.Matrix(res).transpose()

        return final_result

    def calc_coupling_ff(self, uf1, uf2, coupling_matrix):
        """
        Returns the result of the contraction of
        the operator with fermion spinors u1, u2.
        """

        res = uf1.transpose() * coupling_matrix * uf2

        V_simplified = [sp.simplify(component) for component in res]

        return sp.Matrix(V_simplified)

    def calc_coupling_bf(
        self,
        P_f1,
        b1,
        uf2,
        coupling_matrix,
        antispinor=False,
        representation="Lorentz",
        propagator=True,
    ):
        """
        Returns the result of the contraction of
        the operator with fermion spinors u1, u2.
        """
        if representation == "Lorentz":
            P_f = P_f1

        elif representation == "lightcone":
            # Calculate P_slash (P_tot)
            P_f = self.SF.inverse_lightcone_rep(P_f1)

        # Calculate P_slash
        P_tot = self.SF.slash(P_f)

        # Define the spinor u or antispinor
        u = []
        if antispinor == False:

            if propagator == True:
                u = (P_tot + self.SF.m * sp.eye(4)) * b1 * coupling_matrix * uf2
            elif propagator == False:
                u = b1 * coupling_matrix * uf2

        else:
            u = uf2.transpose() * coupling_matrix * b1
            if propagator == True:
                u = uf2.transpose() * coupling_matrix * (-P_tot + self.SF.m * sp.eye(4)) * b1
            elif propagator == False:
                u = uf2.transpose() * coupling_matrix * b1

        u_simplified = [sp.simplify(component) for component in u]

        return sp.Matrix(u_simplified)

    def HG_operator_gg(
        self,
        g1,
        g2,
        p1,
        p2,
        representation="Lorentz",
    ):
        """
        Returns the result of the contraction of
        the operator HG with gluon vector fields g1, g2.
        """

        if representation == "Lorentz":
            P1 = p1  # .transpose() * self.SF.metric()
            P2 = p2  # .transpose() * self.SF.metric()
            g1 = g1
            g2 = g2

        elif representation == "lightcone":

            P1 = self.SF.inverse_lightcone_rep(p1)
            P2 = self.SF.inverse_lightcone_rep(p2)
            g1 = self.SF.inverse_lightcone_rep(g1)
            g2 = self.SF.inverse_lightcone_rep(g2)

        sum = 0
        p1_p2 = P1.transpose() * self.SF.metric() * P2
        g1_P2 = g1.transpose() * self.SF.metric() * P2
        P1_g2 = P1.transpose() * self.SF.metric() * g2

        term1 = sp.Matrix([g1_P2 * P1_g2])  # Dot product or element-wise multiplication
        term2 = p1_p2[0, 0] * g1.transpose() * self.SF.metric() * g2

        sum = 4 * I * self.SF.v * self.SF.C_HG * (term1 - term2)

        scalar_output_current = sum

        return scalar_output_current

    def HG_operator_Hg(
        self,
        g1,
        H,
        p1,
        p2,
        representation="Lorentz",
    ):
        """
        Returns the result of the contraction of
        the operator HG with gluon vector fields g1, g2.
        """

        if representation == "Lorentz":
            P1 = p1  # .transpose() * self.SF.metric()
            P2 = p2  # .transpose() * self.SF.metric()
            g1 = g1

        elif representation == "lightcone":

            P1 = self.SF.four_vector_light_cone(p1)
            P2 = self.SF.four_vector_light_cone(p2)
            g1 = self.SF.four_vector_light_cone(g1)

        sum = 0
        p1_p2 = P1.transpose() * self.SF.metric() * P2
        g1_P2 = g1.transpose() * self.SF.metric() * P2

        term1 = g1_P2[0, 0] * P1  # Dot product or element-wise multiplication
        term2 = p1_p2[0, 0] * g1.transpose() * self.SF.metric()

        g2 = 4 * I * self.SF.v * self.SF.C_HG * (term1 - term2.transpose()) * H

        g2_simplified = [sp.simplify(component) for component in g2]

        if representation == "lightcone":
            g2_simplified = self.SF.inverse_lightcone_rep(g2_simplified)

        return sp.Matrix(g2_simplified)
