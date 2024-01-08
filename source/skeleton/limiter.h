//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  namespace Skeleton
  {
    /**
     * The convex limiter.
     *
     * @ingroup SkeletonEquations
     */
    template <int dim, typename Number = double>
    class Limiter
    {
    public:
      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;

      /**
       * @copydoc HyperbolicSystem::View::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystem::View::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystem::View::flux_contribution_type
       */
      using flux_contribution_type =
          typename HyperbolicSystemView::flux_contribution_type;

      /**
       * @copydoc HyperbolicSystem::View::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(js, U_j, flux_j, scaled_c_ij, beta_ij);
       *   }
       *   limiter.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 0;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const MultiComponentVector<ScalarNumber, n_precomputed_values>
                  &precomputed_values,
              const ScalarNumber relaxation_factor,
              const ScalarNumber newton_tolerance,
              const unsigned int newton_max_iter)
          : hyperbolic_system(hyperbolic_system)
          , precomputed_values(precomputed_values)
          , relaxation_factor(relaxation_factor)
          , newton_tolerance(newton_tolerance)
          , newton_max_iter(newton_max_iter)
      {
      }

      /**
       * Reset temporary storage
       */
      void reset(const unsigned int /*i*/,
                 const state_type & /*new_U_i*/,
                 const flux_contribution_type & /*new_flux_i*/)
      {
        // empty
      }

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int * /*js*/,
                      const state_type & /*U_j*/,
                      const flux_contribution_type & /*flux_j*/,
                      const dealii::Tensor<1, dim, Number> & /*scaled_c_ij*/,
                      const Number & /*beta_ij*/)
      {
        // empty
      }

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number /*hd_i*/) const
      {
        auto relaxed_bounds = bounds_;

        return relaxed_bounds;
      }

      //*}
      /** @name Convex limiter */
      //@{

      /**
       * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
       * function computes and returns the maximal coefficient \f$t\f$,
       * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
       * selected local minimum principles are obeyed.
       */
      std::tuple<Number, bool> limit(const Bounds & /*bounds*/,
                                     const state_type & /*U*/,
                                     const state_type & /*P*/,
                                     const Number /*t_min*/ = Number(0.),
                                     const Number t_max = Number(1.))
      {
        return {t_max, true};
      }

      //*}
      /**
       * @name Verify invariant domain property
       */
      //@{

      /**
       * Returns whether the state @p U is located in the invariant domain
       * described by @p bounds. If @p U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystemView & /*hyperbolic_system*/,
                             const Bounds & /*bounds*/,
                             const state_type & /*U*/)
      {
        return true;
      }

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{
      const HyperbolicSystemView hyperbolic_system;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;

      ScalarNumber relaxation_factor;
      ScalarNumber newton_tolerance;
      unsigned int newton_max_iter;

      Bounds bounds_;
      //@}
    };
  } // namespace Skeleton
} // namespace ryujin
