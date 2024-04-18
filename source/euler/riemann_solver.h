//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace Euler
  {
    template <typename ScalarNumber = double>
    class RiemannSolverParameters : public dealii::ParameterAcceptor
    {
    public:
      RiemannSolverParameters(const std::string &subsection = "/RiemannSolver")
          : ParameterAcceptor(subsection)
      {
        if constexpr (std::is_same<ScalarNumber, double>::value)
          newton_tolerance_ = 1.e-10;
        else
          newton_tolerance_ = 1.e-4;
        add_parameter("newton tolerance",
                      newton_tolerance_,
                      "Tolerance for the quadratic newton stopping criterion");

        newton_max_iterations_ = 0;
        add_parameter("newton max iterations",
                      newton_max_iterations_,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");
      }

      ACCESSOR_READ_ONLY(newton_tolerance);
      ACCESSOR_READ_ONLY(newton_max_iterations);

    private:
      ScalarNumber newton_tolerance_;
      unsigned int newton_max_iterations_;
    };


    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite GuermondPopov2016b.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystemView
       */
      using View = HyperbolicSystemView<dim, Number>;

      /**
       * @copydoc HyperbolicSystemView::problem_dimension
       */
      static constexpr unsigned int problem_dimension = View::problem_dimension;

      /**
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, a]\f$, thus, 4.
       */
      static constexpr unsigned int riemann_data_size = 4;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, a]\f$
       */
      using primitive_type = std::array<Number, riemann_data_size>;

      /**
       * @copydoc HyperbolicSystemView::state_type
       */
      using state_type = typename View::state_type;

      /**
       * @copydoc HyperbolicSystemView::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          View::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystemView::ScalarNumber
       */
      using ScalarNumber = typename View::ScalarNumber;

      /**
       * @copydoc RiemannSolverParameters
       */
      using Parameters = RiemannSolverParameters<ScalarNumber>;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const Parameters &parameters,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and riemann_data_j,
       * compute an estimation of an upper bound for the maximum wavespeed
       * lambda.
       */
      Number compute(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       *
       * Returns a tuple consisting of lambda max and the number of Newton
       * iterations used in the solver to find it.
       */
      Number compute(const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      //@}

    protected:
      /** @name Internal functions used in the Riemann solver */
      //@{

      /**
       * See @cite GuermondPopov2016b, page 912, (3.4).
       *
       * Cost: 1x pow, 1x division, 2x sqrt
       */
      Number f(const primitive_type &riemann_data, const Number p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.4).
       *
       * Cost: 1x pow, 3x division, 1x sqrt
       */
      Number df(const primitive_type &riemann_data, const Number &p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * Cost: 2x pow, 6x division, 2x sqrt
       */
      Number phi(const primitive_type &riemann_data_i,
                 const primitive_type &riemann_data_j,
                 const Number p_in) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * Cost: 2x pow, 6x division, 2x sqrt
       */
      Number dphi(const primitive_type &riemann_data_i,
                  const primitive_type &riemann_data_j,
                  const Number &p) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.3).
       *
       * The approximate Riemann solver is based on a function phi(p) that is
       * montone increasing in p, concave down and whose (weak) third
       * derivative is non-negative and locally bounded [1, p. 912]. Because
       * we actually do not perform any iteration for computing our wavespeed
       * estimate we can get away by only implementing a specialized variant
       * of the phi function that computes phi(p_max). It inlines the
       * implementation of the "f" function and eliminates all unnecessary
       * branches in "f".
       *
       * Cost: 0x pow, 2x division, 2x sqrt
       */
      Number phi_of_p_max(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.7)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda1_minus(const primitive_type &riemann_data,
                           const Number p_star) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.8)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda3_plus(const primitive_type &primitive_state,
                          const Number p_star) const;


      /**
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
       * compute the gap in lambda between both guesses.
       *
       * See @cite GuermondPopov2016b, page 914, (4.4a), (4.4b), (4.5), and
       * (4.6)
       *
       * Cost: 0x pow, 4x division, 4x sqrt
       */
      std::array<Number, 2> compute_gap(const primitive_type &riemann_data_i,
                                        const primitive_type &riemann_data_j,
                                        const Number p_1,
                                        const Number p_2) const;


      /**
       * see @cite GuermondPopov2016b, page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 2x division, 2x sqrt (inclusive)
       */
      Number compute_lambda(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j,
                            const Number p_star) const;


      /**
       * Two-rarefaction approximation to p_star computed for two primitive
       * states <code>riemann_data_i</code> and <code>riemann_data_j</code>.
       *
       * See @cite GuermondPopov2016b, page 914, (4.3)
       *
       * Cost: 2x pow, 2x division, 0x sqrt
       */
      Number p_star_two_rarefaction(const primitive_type &riemann_data_i,
                                    const primitive_type &riemann_data_j) const;

      /**
       * Failsafe approximation to p_star computed for two primitive states
       * <code>riemann_data_i</code> and <code>riemann_data_j</code>.
       *
       * See @cite ClaytonGuermondPopov-2022, (5.11):
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      Number p_star_failsafe(const primitive_type &riemann_data_i,
                             const primitive_type &riemann_data_j) const;


      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      primitive_type
      riemann_data_from_state(const state_type &U,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
      //@}
    };
  } // namespace Euler
} // namespace ryujin
