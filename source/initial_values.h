//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "initial_state_library.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace ryujin
{
  /**
   * A class implementing a number of different initial value
   * configurations.
   *
   * Given a position @p point the member function
   * InitialValues::initial_state() returns the corresponding (conserved)
   * initial state. The function is used to interpolate initial values and
   * enforce Dirichlet boundary conditions. For the latter, the the
   * function signature has an additional parameter @p t denoting the
   * current time to allow for time-dependent (in-flow) Dirichlet data.
   *
   * For validation purposes a number of analytic solutions are implemented
   * as well.
   *
   * @ingroup InitialValues
   */
  template <typename Description, int dim, typename Number = double>
  class InitialValues : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc HyperbolicSystemView
     */
    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension = View::problem_dimension;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = typename View::state_type;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * @copydoc HyperbolicSystem::n_precomputed_values
     */
    static constexpr unsigned int n_precomputed_values =
        View::n_precomputed_initial_values;

    /**
     * Array type used for precomputed values.
     */
    using precomputed_state_type = typename View::precomputed_state_type;

    /**
     * Constructor.
     */
    InitialValues(const HyperbolicSystem &hyperbolic_system,
                  const OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "/InitialValues");


    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some initial
     * state values. Do this with a callback.
     */
    void parse_parameters_callback();


    /**
     * Given a position @p point returns the corresponding (conserved)
     * initial state. The function is used to interpolate initial values
     * and enforce Dirichlet boundary conditions. For the latter, the the
     * function signature has an additional parameter @p t denoting the
     * current time to allow for time-dependent (in-flow) Dirichlet data.
     */
    DEAL_II_ALWAYS_INLINE inline state_type
    initial_state(const dealii::Point<dim> &point, Number t) const
    {
      return initial_state_(point, t);
    }


    /**
     * This routine computes and returns a state vector populated with
     * initial values for a specified time @p t.
     */
    vector_type interpolate(Number t = 0) const;


    /**
     * Given a position @p point returns the corresponding (conserved)
     * initial state. The function is used to interpolate initial values
     * and enforce Dirichlet boundary conditions. For the latter, the the
     * function signature has an additional parameter @p t denoting the
     * current time to allow for time-dependent (in-flow) Dirichlet data.
     */
    DEAL_II_ALWAYS_INLINE inline precomputed_state_type
    flux_contributions(const dealii::Point<dim> &point) const
    {
      return flux_contributions_(point);
    }


    /**
     * This routine computes and returns a state vector populated with
     * initial values for a specified time @p t.
     */
    MultiComponentVector<Number, n_precomputed_values>
    interpolate_precomputed_initial_values() const;

  private:
    /**
     * @name Run time options
     */
    //@{

    std::string configuration_;

    dealii::Point<dim> initial_position_;

    dealii::Tensor<1, dim> initial_direction_;

    Number perturbation_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    typename InitialStateLibrary<Description, dim, Number>::
        initial_state_list_type initial_state_list_;

    std::function<state_type(const dealii::Point<dim> &point, Number t)>
        initial_state_;

    std::function<precomputed_state_type(const dealii::Point<dim> &point)>
        flux_contributions_;

    //@}
  };

} /* namespace ryujin */
