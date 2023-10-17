//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
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

#include "level_structures.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>

#include <fstream>
#include <future>
#include <sstream>
namespace ryujin{


  namespace mgrit
  {

    /**
     * The high-level time loop driving the computation, this one is specific to the
     * MGRIT algorithm.
     *
     * @ingroup TimeLoop
     */
    template <typename Description, int dim, typename Number = double>
    class TimeLoopMgrit final : public dealii::ParameterAcceptor
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
         * @copydoc HyperbolicSystem::View
         */
        using HyperbolicSystemView =
            typename Description::HyperbolicSystem::template View<dim, Number>;

        /**
         * @copydoc HyperbolicSystem::problem_dimension
         */
        static constexpr unsigned int problem_dimension =
            HyperbolicSystemView::problem_dimension;

        /**
         * @copydoc HyperbolicSystem::n_precomputed_values
         */
        static constexpr unsigned int n_precomputed_values =
            HyperbolicSystemView::n_precomputed_values;


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
        TimeLoopMgrit(const MPI_Comm &mpi_comm,
                      const LevelStructures<Description, dim, Number> &ls,
                      const Number initial_time,
                      const Number final_time);

        /**
         * Run the high-level time loop.
         */
        void run();

        void run_with_initial_data(const vector_type &U);

        //returns the solution state U, if the size is greater than one.
        vector_type get_U();

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

        std::string base_name_;

        Number t_initial_;
        Number t_final_;
        vector_type U_;

        std::vector<Number> t_refinements_;

        Number output_granularity_;

        bool enable_checkpointing_;
        bool enable_output_full_;
        bool enable_output_levelsets_;
        bool enable_compute_error_;
        bool enable_compute_quantities_;
        bool post_process_;

        unsigned int output_checkpoint_multiplier_;
        unsigned int output_full_multiplier_;
        unsigned int output_levelsets_multiplier_;
        unsigned int output_quantities_multiplier_;

        std::vector<std::string> error_quantities_;
        bool error_normalize_;

        bool resume_;

        Number terminal_update_interval_;
        bool terminal_show_rank_throughput_;

        //@}
        /**
         * @name Internal data:
         */
        //@{

        const MPI_Comm &mpi_communicator_;

        using map_type = typename std::map<std::string, dealii::Timer>;
        map_type computing_timer_;

        //todo: add pointers here.
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

  } // namespace mgrit
}//namespace ryujin
