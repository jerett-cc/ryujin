//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "hyperbolic_module.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>

#include <fstream>
#include <future>
#include <sstream>

#include <functional>
#include <level_structures.h>

namespace ryujin
{

  /**
   * The high-level time loop driving the computation.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class TimeLoop final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc ParabolicSystem
     */
    using ParabolicSystem = typename Description::ParabolicSystem;

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
     * @copydoc HyperbolicSystem::n_precomputed_values
     */
    static constexpr unsigned int n_precomputed_values =
        View::n_precomputed_values;


    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Typedef for a MultiComponentVector storing precomputed values.
     */
    using precomputed_type = MultiComponentVector<Number, n_precomputed_values>;

    /**
     * Constructor.
     */
    TimeLoop(const MPI_Comm &mpi_comm);

    /**
     * Constructor with LevelStructures Object. 
     */
    TimeLoop(const MPI_Comm &mpi_comm, const mgrit::LevelStructures<Description, dim, Number> &ls);

    /**
     * Run the high-level time loop.
     */
    void run();

    /**
     * Run the high-level time loop, with a reference to existing data. 
     * Optional postprocess function which needs to take the vector and the current time.
     */
    void run_with_initial_data(vector_type &U, const Number end_time, const Number start_time=0, 
                               std::function<void(const vector_type&/*U*/,double/*current time*/)> pp_step = std::function<void(const vector_type&,double)>{});

  protected:
    /**
     * @name Private methods for run()
     */
    //@{

    void compute_error(const vector_type &U, Number t);

    void output(const vector_type &U,
                const std::string &name,
                Number t,
                unsigned int cycle);

    void print_parameters(std::ostream &stream);
    void print_mpi_partition(std::ostream &stream);
    void print_memory_statistics(std::ostream &stream);
    void print_timers(std::ostream &stream);
    void print_throughput(unsigned int cycle,
                          Number t,
                          std::ostream &stream,
                          bool final_time = false);

    void print_info(const std::string &header);
    void print_head(const std::string &header,
                    const std::string &secondary,
                    std::ostream &stream);

    void print_cycle_statistics(unsigned int cycle,
                                Number t,
                                unsigned int output_cycle,
                                bool write_to_logfile = false,
                                bool final_time = false);
    //@}

  private:
    /**
     * @name Run time options
     */
    //@{
    
    void declare_parameters();

    std::string base_name_;

    Number t_final_;
    std::vector<Number> t_refinements_;

    Number output_granularity_;

    bool enable_checkpointing_;
    bool enable_output_full_;
    bool enable_output_levelsets_;
    bool enable_compute_error_;
    bool enable_compute_quantities_;

    unsigned int output_checkpoint_multiplier_;
    unsigned int output_full_multiplier_;
    unsigned int output_levelsets_multiplier_;
    unsigned int output_quantities_multiplier_;

    std::vector<std::string> error_quantities_;
    bool error_normalize_;

    bool resume_;
    bool resume_at_time_zero_;

    Number terminal_update_interval_;
    bool terminal_show_rank_throughput_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    std::map<std::string, dealii::Timer> computing_timer_;

    std::shared_ptr<HyperbolicSystem> hyperbolic_system_;
    std::shared_ptr<ParabolicSystem> parabolic_system_;
    std::shared_ptr<Discretization<dim>> discretization_;
    std::shared_ptr<OfflineData<dim, Number>> offline_data_;
    std::shared_ptr<InitialValues<Description, dim, Number>> initial_values_;
    std::shared_ptr<HyperbolicModule<Description, dim, Number>> hyperbolic_module_;
    std::shared_ptr<ParabolicModule<Description, dim, Number>> parabolic_module_;
    std::shared_ptr<TimeIntegrator<Description, dim, Number>> time_integrator_;
    std::shared_ptr<Postprocessor<Description, dim, Number>> postprocessor_;
    std::shared_ptr<VTUOutput<Description, dim, Number>> vtu_output_;
    std::shared_ptr<Quantities<Description, dim, Number>> quantities_;

    const unsigned int mpi_rank_;
    const unsigned int n_mpi_processes_;

    std::ofstream logfile_; /* log file */

    //@}
  };

} // namespace ryujin
