//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "patterns_conversion.h"

#include <limits>

namespace ryujin
{
  /**
   * Controls the chosen invariant domain / CFL recovery strategy.
   *
   * @ingroup TimeLoop
   */
  enum class CFLRecoveryStrategy {
    /**
     * Step with the chosen "cfl max" value and do nothing in case an
     * invariant domain and or CFL condition violation is detected.
     */
    none,

    /**
     * Step with the chosen "cfl max" value and, in case an invariant
     * domain and or CFL condition violation is detected, the time step
     * is repeated with "cfl min". If this is unsuccessful as well, a
     * warning is emitted.
     */
    bang_bang_control,
  };


  /**
   * Controls the chosen time-stepping scheme.
   *
   * @ingroup TimeLoop
   */
  enum class TimeSteppingScheme {
    /**
     * The strong stability preserving Runge Kutta method of order 2,
     * SSPRK(2,2;1/2), with the following butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{2} & \tfrac{1}{2} & 0 \\
     *   \hline
     *   1            & 1  & 0
     * \end{array}
     * \f}
     */
    ssprk_22,
    /**
     * The strong stability preserving Runge Kutta method of order 3,
     * SSPRK(3,3;1/3), with the following butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   1            & 1            & 0 \\
     *   \tfrac{1}{2} & \tfrac{1}{4} & \tfrac{1}{4} & 0\\
     *   \hline
     *   1            & \tfrac{1}{6} & \tfrac{1}{6} & \tfrac{2}{3}
     * \end{array}
     * \f}
     */
    ssprk_33,

    /**
     * The explicit Runge-Kutta method RK(1,1;1), aka a simple, forward
     * Euler step.
     */
    erk_11,

    /**
     * The explicit Runge-Kutta method RK(2,2;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{2} & \tfrac{1}{2} & 0 \\
     *   \hline
     *   1            & 0  & 1
     * \end{array}
     * \f}
     */
    erk_22,

    /**
     * The explicit Runge-Kutta method RK(3,3;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{3} & \tfrac{1}{3} & 0 \\
     *   \tfrac{2}{3} & 0            & \tfrac{2}{3} & 0 \\
     *   \hline
     *   1            & \tfrac{1}{4} & 0            & \tfrac{3}{4}
     * \end{array}
     * \f}
     */
    erk_33,

    /**
     * The explicit Runge-Kutta method RK(4,3;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{4} & \tfrac{1}{4} & 0 \\
     *   \tfrac{1}{2} & 0            & \tfrac{1}{2} & 0 \\
     *   \tfrac{3}{4} & 0            & \tfrac{1}{4} & \tfrac{1}{2}  & 0 \\
     *   \hline
     *   1            & 0            & \tfrac{2}{3} & -\tfrac{1}{3} &
     * \tfrac{2}{3} \end{array} \f}
     */
    erk_43,

    /**
     * The explicit Runge-Kutta method RK(5,4;1) with the butcher tableau
     * TODO
     */
    erk_54,

    /**
     * A Strang split using ssprk 33 for the hyperbolic subproblem and
     * Crank-Nicolson for the parabolic subproblem
     */
    strang_ssprk_33_cn,

    /**
     * A Strang split using erk 33 for the hyperbolic subproblem and
     * Crank-Nicolson for the parabolic subproblem
     */
    strang_erk_33_cn,

    /**
     * A Strang split using erk 43 for the hyperbolic subproblem and
     * Crank-Nicolson for the parabolic subproblem
     */
    strang_erk_43_cn,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::CFLRecoveryStrategy,
             LIST({ryujin::CFLRecoveryStrategy::none, "none"},
                  {ryujin::CFLRecoveryStrategy::bang_bang_control,
                   "bang bang control"}));

DECLARE_ENUM(
    ryujin::TimeSteppingScheme,
    LIST({ryujin::TimeSteppingScheme::ssprk_22, "ssprk 22"},
         {ryujin::TimeSteppingScheme::ssprk_33, "ssprk 33"},
         {ryujin::TimeSteppingScheme::erk_11, "erk 11"},
         {ryujin::TimeSteppingScheme::erk_22, "erk 22"},
         {ryujin::TimeSteppingScheme::erk_33, "erk 33"},
         {ryujin::TimeSteppingScheme::erk_43, "erk 43"},
         {ryujin::TimeSteppingScheme::erk_54, "erk 54"},
         {ryujin::TimeSteppingScheme::strang_ssprk_33_cn, "strang ssprk 33 cn"},
         {ryujin::TimeSteppingScheme::strang_erk_33_cn, "strang erk 33 cn"},
         {ryujin::TimeSteppingScheme::strang_erk_43_cn, "strang erk 43 cn"}, ));
#endif

namespace ryujin
{
  /**
   * The TimeIntegrator class implements IMEX timestepping strategies based
   * on explicit and diagonally-implicit Runge Kutta schemes.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class TimeIntegrator final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using ParabolicSystem = typename Description::ParabolicSystem;

    using StateVector = typename View::StateVector;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    TimeIntegrator(
        const MPI_Comm &mpi_communicator,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicModule<Description, dim, Number> &hyperbolic_module,
        const ParabolicModule<Description, dim, Number> &parabolic_module,
        const std::string &subsection = "/TimeIntegrator");

    /**
     * Prepare time integration. A call to prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    //@}
    /**
     * @name Functions for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U performs an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * @note This function switches between different Runge-Kutta methods
     * depending on chosen runtime parameters.
     *
     * @note Depending on chosen run time parameters different CFL
     * adaptation and recovery strategies for invariant domain violations
     * are used.
     */
    Number step(StateVector &state_vector, Number t, const Number t_final = -1.0);

    /**
     * The selected time-stepping scheme.
     */
    ACCESSOR_READ_ONLY(time_stepping_scheme);

    /**
     * The eficiency of the selected time-stepping scheme expressed as the
     * ratio of step size of the combined method to step size of an
     * elementary forward Euler step. For example, SSPRK33 has an
     * efficiency ratio of 1 whereas ERK33 has an efficiency ratio of 3.
     */
    ACCESSOR_READ_ONLY(efficiency);

  protected:
    /**
     * Given a reference to a previous state vector U performs an explicit
     * second-order strong-stability preserving Runge-Kutta SSPRK(2,2;1/2)
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * If the parameter @p tau is set to a nonzero value then the
     * supplied value is used for time stepping instead of the computed
     * maximal time step size.
     */
    Number step_ssprk_22(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order strong-stability preserving Runge-Kutta SSPRK(3,3;1/3)
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * If the parameter @p tau is set to a nonzero value then the
     * supplied value is used for time stepping instead of the computed
     * maximal time step size.
     */
    Number step_ssprk_33(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * first-order Euler step ERK(1,1;1) time step (and store the result
     * in U). The function returns the chosen time step size tau.
     */
    Number step_erk_11(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * second-order Runge-Kutta ERK(2,2;1) time step (and store the result
     * in U). The function returns the chosen time step size tau.
     */
    Number step_erk_22(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order Runge-Kutta ERK(3,3;1) time step (and store the result
     * in U). The function returns the chosen time step size tau.
     */
    Number step_erk_33(StateVector &state_vector, Number t, const Number t_final);//TODO: make a default here?

    /**
     * Given a reference to a previous state vector U performs an explicit
     * 4 stage third-order Runge-Kutta ERK(4,3;1) time step (and store the
     * result in U). The function returns the chosen time step size tau.
     */
    Number step_erk_43(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * 4 stage fourth-order Runge-Kutta ERK(5,4;1) time step (and store
     * the result in U). The function returns the chosen time step size
     * tau.
     */
    Number step_erk_54(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs a combined
     * explicit implicit Strang split using a third-order Runge-Kutta
     * ERK(3,3;1/3) time step and an implicit Crank-Nicolson step (and
     * store the result in U). The function returns the chosen time step
     * size tau.
     */
    Number step_strang_ssprk_33_cn(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs a combined
     * explicit implicit Strang split using a third-order Runge-Kutta
     * ERK(3,3;1) time step and an implicit Crank-Nicolson step (and store
     * the result in U). The function returns the chosen time step size
     * tau.
     */
    Number step_strang_erk_33_cn(StateVector &state_vector, Number t);

    /**
     * Given a reference to a previous state vector U performs a combined
     * explicit implicit Strang split using a third-order Runge-Kutta
     * ERK(4,3;1) time step and an implicit Crank-Nicolson step (and store
     * the result in U). The function returns the chosen time step size
     * tau.
     */
    Number step_strang_erk_43_cn(StateVector &state_vector, Number t);

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Number cfl_min_;
    Number cfl_max_;

    CFLRecoveryStrategy cfl_recovery_strategy_;

    TimeSteppingScheme time_stepping_scheme_;
    double efficiency_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicModule<Description, dim, Number>>
        hyperbolic_module_;
    dealii::SmartPointer<const ParabolicModule<Description, dim, Number>>
        parabolic_module_;

    std::vector<StateVector> temp_;

    //@}
  };

} /* namespace ryujin */
