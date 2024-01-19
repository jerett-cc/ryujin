//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "checkpointing.h"
#include "introspection.h"
#include "scope.h"
#include "solution_transfer.h"
#include "time_loop_mgrit.h"
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "geometry_cylinder.h"
#include "discretization.h"
#include "euler/parabolic_system.h"
#include "euler/description.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"
#include "level_structures.h"


#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/tria_accessor.h>
#include <vector>

#include <fstream>
#include <iomanip>

using namespace dealii;

namespace ryujin{
  namespace mgrit
  {

    template <typename Description, int dim, typename Number>
    TimeLoopMgrit<Description, dim, Number>::TimeLoopMgrit(const MPI_Comm &mpi_comm,
                                                           const LevelStructures<Description,dim, Number> &ls,
                                                           const Number initial_time,
                                                           const Number final_time)
    : ParameterAcceptor("/TimeLoop")
    , mpi_communicator_(mpi_comm)
    , hyperbolic_system_(ls.hyperbolic_system)
    , parabolic_system_(ls.parabolic_system)
    , discretization_(ls.discretization)
    , offline_data_(ls.offline_data)
    , initial_values_(ls.initial_values)
    , hyperbolic_module_(ls.hyperbolic_module)
    , parabolic_module_(ls.parabolic_module)
    , time_integrator_(ls.time_integrator)
    , postprocessor_(ls.postprocessor)
    , vtu_output_(ls.vtu_output)
    , quantities_(ls.quantities)
    , mpi_rank_(dealii::Utilities::MPI::this_mpi_process(mpi_communicator_))
    , n_mpi_processes_(
        dealii::Utilities::MPI::n_mpi_processes(mpi_communicator_))
    {
      base_name_ = "cylinder";
      add_parameter("basename", base_name_, "Base name for all output files");

      t_initial_ = initial_time;

      t_final_ = final_time;

      add_parameter("refinement timepoints",
          t_refinements_,
          "List of points in (simulation) time at which the mesh will "
          "be globally refined");

      output_granularity_ = Number(1);
      add_parameter(
          "output granularity",
          output_granularity_,
          "The output granularity specifies the time interval after which output "
          "routines are run. Further modified by \"*_multiplier\" options");

      enable_checkpointing_ = false;
      add_parameter(
          "enable checkpointing",
          enable_checkpointing_,
          "Write out checkpoints to resume an interrupted computation "
          "at output granularity intervals. The frequency is determined by "
          "\"output granularity\" times \"output checkpoint multiplier\"");

      enable_output_full_ = true;
      add_parameter("enable output full",
          enable_output_full_,
          "Write out full pvtu records. The frequency is determined by "
          "\"output granularity\" times \"output full multiplier\"");

      enable_output_levelsets_ = true;
      add_parameter(
          "enable output levelsets",
          enable_output_levelsets_,
          "Write out levelsets pvtu records. The frequency is determined by "
          "\"output granularity\" times \"output levelsets multiplier\"");

      enable_compute_error_ = false;
      add_parameter("enable compute error",
          enable_compute_error_,
          "Flag to control whether we compute the Linfty Linf_norm of "
          "the difference to an analytic solution. Implemented only "
          "for certain initial state configurations.");

      enable_compute_quantities_ = false;
      add_parameter(
          "enable compute quantities",
          enable_compute_quantities_,
          "Flag to control whether we compute quantities of interest. The "
          "frequency how often quantities are logged is determined by \"output "
          "granularity\" times \"output quantities multiplier\"");

      output_checkpoint_multiplier_ = 1;
      add_parameter("output checkpoint multiplier",
          output_checkpoint_multiplier_,
          "Multiplicative modifier applied to \"output granularity\" "
          "that determines the checkpointing granularity");

      output_full_multiplier_ = 1;
      add_parameter("output full multiplier",
          output_full_multiplier_,
          "Multiplicative modifier applied to \"output granularity\" "
          "that determines the full pvtu writeout granularity");

      output_levelsets_multiplier_ = 1;
      add_parameter("output levelsets multiplier",
          output_levelsets_multiplier_,
          "Multiplicative modifier applied to \"output granularity\" "
          "that determines the levelsets pvtu writeout granularity");

      output_quantities_multiplier_ = 1;
      add_parameter(
          "output quantities multiplier",
          output_quantities_multiplier_,
          "Multiplicative modifier applied to \"output granularity\" that "
          "determines the writeout granularity for quantities of interest");

      std::copy(std::begin(HyperbolicSystemView::component_names),
          std::end(HyperbolicSystemView::component_names),
          std::back_inserter(error_quantities_));

      add_parameter("error quantities",
          error_quantities_,
          "List of conserved quantities used in the computation of the "
          "error norms.");

      error_normalize_ = true;
      add_parameter("error normalize",
          error_normalize_,
          "Flag to control whether the error should be normalized by "
          "the corresponding norm of the analytic solution.");

      resume_ = false;
      add_parameter("resume", resume_, "Resume an interrupted computation");

      terminal_update_interval_ = 5;
      add_parameter("terminal update interval",
          terminal_update_interval_,
          "Number of seconds after which output statistics are "
          "recomputed and printed on the terminal");

      terminal_show_rank_throughput_ = true;
      add_parameter("terminal show rank throughput",
          terminal_show_rank_throughput_,
          "If set to true an average per rank throughput is computed "
          "by dividing the total consumed CPU time (per rank) by the "
          "number of threads (per rank). If set to false then a plain "
          "average per thread \"CPU\" throughput value is computed by "
          "using the umodified total accumulated CPU time.");
    }

    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::run()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "TimeLoopMgrit<dim, Number>::run()" << std::endl;
#endif

      const bool write_output_files = enable_checkpointing_ ||
          enable_output_full_ ||
          enable_output_levelsets_;

      /* Attach log file: */
      if (mpi_rank_ == 0)
        logfile_.open(base_name_ + ".log");

      print_parameters(logfile_);

      Number t = 0.;
      unsigned int output_cycle = 0;
      vector_type U;

      /* Prepare data structures: */

      const auto prepare_compute_kernels = [&]() {
        offline_data_->prepare(problem_dimension);
        hyperbolic_module_->prepare();
        parabolic_module_->prepare();
        time_integrator_->prepare();
        postprocessor_->prepare();
        vtu_output_->prepare();
        /* We skip the first output cycle for quantities: */
        quantities_->prepare(base_name_, output_cycle == 0 ? 1 : output_cycle);
        print_mpi_partition(logfile_);
      };

      {
        Scope scope(computing_timer_, "(re)initialize data structures");
        print_info("initializing data structures");

        if (resume_) {
          print_info("resuming computation: recreating mesh");
          Checkpointing::load_mesh(*discretization_, base_name_);

//          print_info("preparing compute kernels");
//          prepare_compute_kernels();todo: remove

          print_info("resuming computation: loading state vector");
          U.reinit(offline_data_->vector_partitioner());
          Checkpointing::load_state_vector(
              *offline_data_, base_name_, U, t, output_cycle, mpi_communicator_);
          t_initial_ = t;

          /* Workaround: Reinitialize Quantities with correct output cycle: */
          quantities_->prepare(base_name_, output_cycle);

          /* Remove outdated refinement timestamps: */
          const auto new_end =
              std::remove_if(t_refinements_.begin(),
                  t_refinements_.end(),
                  [&](const Number &t_ref) { return (t >= t_ref); });
          t_refinements_.erase(new_end, t_refinements_.end());

        } else {

//          print_info("creating mesh");
//          discretization_.prepare();todo remove this

//          print_info("preparing compute kernels");
//          prepare_compute_kernels();todo remove this

          print_info("interpolating initial values");
          U.reinit(offline_data_->vector_partitioner());
          U = initial_values_->interpolate();
#ifdef DEBUG
          /* Poison constrained degrees of freedom: */
          const unsigned int n_relevant = offline_data_->n_locally_relevant();
          const auto &partitioner = offline_data_->scalar_partitioner();
          for (unsigned int i = 0; i < n_relevant; ++i) {
            if (offline_data_->affine_constraints().is_constrained(
                partitioner->local_to_global(i)))
              U.write_tensor(dealii::Tensor<1, dim + 2, Number>() *
                  std::numeric_limits<Number>::signaling_NaN(),
                  i);
          }
#endif
        }
      }

      unsigned int cycle = 1;
      Number last_terminal_output = (terminal_update_interval_ == Number(0.)
          ? std::numeric_limits<Number>::max()
      : std::numeric_limits<Number>::lowest());

      /* Loop: */

      print_info("entering main loop");
      computing_timer_["time loop"].start();


      for (;; ++cycle) {

#ifdef DEBUG_OUTPUT
        std::cout << "\n\n###   cycle = " << cycle << "   ###\n\n" << std::endl;
#endif

        /* Accumulate quantities of interest: */

        if (enable_compute_quantities_) {
          Scope scope(computing_timer_,
              "time step [X] 1 - accumulate quantities");
          quantities_->accumulate(U, t);
        }

        /* Perform output: */

        if (t >= output_cycle * output_granularity_) {
          if (write_output_files) {
            output(U, base_name_ + "-solution", t, output_cycle);
            if (enable_compute_error_) {
              const auto analytic = initial_values_->interpolate(t);
              output(
                  analytic, base_name_ + "-analytic_solution", t, output_cycle);
            }
          }
          if (enable_compute_quantities_ &&
              (output_cycle % output_quantities_multiplier_ == 0) &&
              (output_cycle > 0)) {
            Scope scope(computing_timer_,
                "time step [X] 2 - write out quantities");
            quantities_->write_out(U, t, output_cycle);
          }
          ++output_cycle;
        }

        /* Perform global refinement: */

        const auto new_end = std::remove_if(
            t_refinements_.begin(),
            t_refinements_.end(),
            [&](const Number &t_ref) {
          if (t < t_ref)
            return false;

          computing_timer_["time loop"].stop();
          Scope scope(computing_timer_, "(re)initialize data structures");

          print_info("performing global refinement");

          SolutionTransfer<Description, dim, Number> solution_transfer(
              *offline_data_, *hyperbolic_system_);

          auto &triangulation = discretization_->triangulation();
          for (auto &cell : triangulation.active_cell_iterators())
            cell->set_refine_flag();
          triangulation.prepare_coarsening_and_refinement();

          solution_transfer.prepare_for_interpolation(U);

          triangulation.execute_coarsening_and_refinement();
          prepare_compute_kernels();

          solution_transfer.interpolate(U);

          computing_timer_["time loop"].start();
          return true;
        });
        t_refinements_.erase(new_end, t_refinements_.end());

        /* Break if we have reached the final time: */

        if (t >= t_final_)
          break;


        const auto tau = time_integrator_->step(U, t);
        t += tau;

        /* Print and record cycle statistics: */

        const bool write_to_log_file = (t >= output_cycle * output_granularity_);
        const auto wall_time = computing_timer_["time loop"].wall_time();
        const auto data =
            Utilities::MPI::min_max_avg(wall_time, mpi_communicator_);
        const bool update_terminal =
            (data.avg >= last_terminal_output + terminal_update_interval_);
        if (terminal_update_interval_ != Number(0.)) {
          if (write_to_log_file || update_terminal) {
            print_cycle_statistics(
                cycle, t, output_cycle, /*logfile*/ write_to_log_file);
            last_terminal_output = data.avg;
          }
        }
      } /* end of loop */

      //U_ = U;//set the member U to be the solution U calculated above
      /* We have actually performed one cycle less. */
      --cycle;

      computing_timer_["time loop"].stop();

      if (terminal_update_interval_ != Number(0.)) {
        /* Write final timing statistics to screen and logfile: */
        print_cycle_statistics(
            cycle, t, output_cycle, /*logfile*/ true, /*final*/ true);
      }

      if (enable_compute_error_) {
        /* Output final error: */
        compute_error(U, t);
      }


#ifdef WITH_VALGRIND
      CALLGRIND_DUMP_STATS;
#endif
    }

    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::run_with_initial_data(vector_type &U, 
      const Number end_time, const Number start_time, const bool print_vector)
    {

      if(print_vector)
      {
        std::cout << "U before stepping." << std::endl;
        U.print(std::cout);
      }
      // if(dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0)
        // U.print(std::cout);
      const bool write_output_files = enable_checkpointing_ ||
          enable_output_full_ ||
          enable_output_levelsets_;

      Number t = start_time;//we will start at the initial time that this object was created with
      unsigned int output_cycle = 0;

      unsigned int cycle = 1;
      Number last_terminal_output = (terminal_update_interval_ == Number(0.)
          ? std::numeric_limits<Number>::max()
      : std::numeric_limits<Number>::lowest());

      /* Loop: */

      print_info("entering main loop");

      for (;; ++cycle) {

        /* Accumulate quantities of interest: */

        if (enable_compute_quantities_) {
          quantities_->accumulate(U, t);
        }
        if(print_vector)
        {
          std::string fname = "timeloop" + std::to_string(cycle);
          std::cout << "Printing U to file before step in file " << fname << std::endl;
          output(U,fname,t,0);
        }

        /* Break if we have reached the final time: */

        if (t >= end_time)
          break;


        /* Take a step: */
        const auto tau = time_integrator_->step(U, t);
        t += tau;

      } /* end of loop */
      /* We have actually performed one cycle less. */
      --cycle;

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::compute_error(
        const typename TimeLoopMgrit<Description, dim, Number>::vector_type &U,
        const Number t)
    {
#ifdef DEBUG_OUTPUT
      std::cout << "TimeLoopMgrit<dim, Number>::compute_error()" << std::endl;
#endif

      Vector<Number> difference_per_cell(
          discretization_->triangulation().n_active_cells());

      Number linf_norm = 0.;
      Number l1_norm = 0;
      Number l2_norm = 0;

      const auto analytic = initial_values_->interpolate(t);

      scalar_type analytic_component;
      scalar_type error_component;
      analytic_component.reinit(offline_data_->scalar_partitioner());
      error_component.reinit(offline_data_->scalar_partitioner());

      /* Loop over all selected components: */
      for (const auto &entry : error_quantities_) {
        const auto &names = HyperbolicSystemView::component_names;
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos == std::end(names)) {
          AssertThrow(
              false,
              dealii::ExcMessage("Unknown component name »" + entry + "«"));
          __builtin_trap();
        }

        const auto index = std::distance(std::begin(names), pos);

        analytic.extract_component(analytic_component, index);

        /* Compute norms of analytic solution: */

        Number linf_norm_analytic = 0.;
        Number l1_norm_analytic = 0.;
        Number l2_norm_analytic = 0.;

        if (error_normalize_) {
          linf_norm_analytic = Utilities::MPI::max(
              analytic_component.linfty_norm(), mpi_communicator_);

          VectorTools::integrate_difference(
              offline_data_->dof_handler(),
              analytic_component,
              Functions::ZeroFunction<dim, Number>(),
              difference_per_cell,
              QGauss<dim>(3),
              VectorTools::L1_norm);

          l1_norm_analytic = Utilities::MPI::sum(difference_per_cell.l1_norm(),
              mpi_communicator_);

          VectorTools::integrate_difference(
              offline_data_->dof_handler(),
              analytic_component,
              Functions::ZeroFunction<dim, Number>(),
              difference_per_cell,
              QGauss<dim>(3),
              VectorTools::L2_norm);

          l2_norm_analytic = Number(std::sqrt(Utilities::MPI::sum(
              std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator_)));
        }

        /* Compute norms of error: */

        U.extract_component(error_component, index);
        /* Populate constrained dofs due to periodicity: */
        offline_data_->affine_constraints().distribute(error_component);
        error_component.update_ghost_values();
        error_component -= analytic_component;

        const Number linf_norm_error =
            Utilities::MPI::max(error_component.linfty_norm(), mpi_communicator_);

        VectorTools::integrate_difference(offline_data_->dof_handler(),
            error_component,
            Functions::ZeroFunction<dim, Number>(),
            difference_per_cell,
            QGauss<dim>(3),
            VectorTools::L1_norm);

        const Number l1_norm_error =
            Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator_);

        VectorTools::integrate_difference(offline_data_->dof_handler(),
            error_component,
            Functions::ZeroFunction<dim, Number>(),
            difference_per_cell,
            QGauss<dim>(3),
            VectorTools::L2_norm);

        const Number l2_norm_error = Number(std::sqrt(Utilities::MPI::sum(
            std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator_)));

        if (error_normalize_) {
          linf_norm += linf_norm_error / linf_norm_analytic;
          l1_norm += l1_norm_error / l1_norm_analytic;
          l2_norm += l2_norm_error / l2_norm_analytic;
        } else {
          linf_norm += linf_norm_error;
          l1_norm += l1_norm_error;
          l2_norm += l2_norm_error;
        }
      }

      if (mpi_rank_ != 0)
        return;

      logfile_ << std::endl << "Computed errors:" << std::endl << std::endl;
      logfile_ << std::setprecision(16);

      std::string description =
          error_normalize_ ? "Normalized consolidated" : "Consolidated";

      logfile_ << description + " Linf, L1, and L2 errors at final time \n";
      logfile_ << std::setprecision(16);
      logfile_ << "#dofs = " << offline_data_->dof_handler().n_dofs() << std::endl;
      logfile_ << "t     = " << t << std::endl;
      logfile_ << "Linf  = " << linf_norm << std::endl;
      logfile_ << "L1    = " << l1_norm << std::endl;
      logfile_ << "L2    = " << l2_norm << std::endl;

      std::cout << description + " Linf, L1, and L2 errors at final time \n";
      std::cout << std::setprecision(16);
      std::cout << "#dofs = " << offline_data_->dof_handler().n_dofs()
                  << std::endl;
      std::cout << "t     = " << t << std::endl;
      std::cout << "Linf  = " << linf_norm << std::endl;
      std::cout << "L1    = " << l1_norm << std::endl;
      std::cout << "L2    = " << l2_norm << std::endl;
    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::output(
        const typename TimeLoopMgrit<Description, dim, Number>::vector_type &U,
        const std::string &name,
        Number t,
        unsigned int cycle)
    {
#ifdef DEBUG_OUTPUT
      std::cout << "TimeLoopMgrit<dim, Number>::output(t = " << t << ")" << std::endl;
#endif

      const bool do_full_output =
          (cycle % output_full_multiplier_ == 0) && enable_output_full_;
      const bool do_levelsets =
          (cycle % output_levelsets_multiplier_ == 0) && enable_output_levelsets_;
      const bool do_checkpointing =
          (cycle % output_checkpoint_multiplier_ == 0) && enable_checkpointing_;

      // std::cout << "Do full output in output: " << do_full_output << std::endl;
      // std::cout << "output_full_multiplier in output: " << output_full_multiplier_ << std::endl;
      // std::cout << "cycle % outputfullmultiplier in output: " << (cycle%output_full_multiplier_ == 0) << std::endl;
      // std::cout << "enable_output_full in output: " << enable_output_full_ << std::endl;

      /* There is nothing to do: */
      if (!(do_full_output || do_levelsets || do_checkpointing))
        return;

      /* Data output: */
      if (do_full_output || do_levelsets) {
        Scope scope(computing_timer_, "time step [X] 3 - output vtu");
        print_info("scheduling output");

        postprocessor_->compute(U);
        /*
         * Workaround: Manually reset bounds during the first output cycle
         * (which is often just a uniform flow field) to obtain a better
         * normailization:
         */
        if (cycle == 0)
          postprocessor_->reset_bounds();

        precomputed_type precomputed_values;

        if (vtu_output_->need_to_prepare_step()) {
          /*
           * In case we output a precomputed value or alpha we have to run
           * Steps 0 - 2 of the explicit Euler step:
           */
          const auto &scalar_partitioner = offline_data_->scalar_partitioner();
          precomputed_values.reinit_with_scalar_partitioner(scalar_partitioner);

          vector_type dummy;
          hyperbolic_module_->precompute_only_ = true;
          hyperbolic_module_->template step<0>(
              U, {}, {}, {}, dummy, precomputed_values, Number(0.));
          hyperbolic_module_->precompute_only_ = false;
        }

        std::cout << "Scheduling output" << std::endl;
        vtu_output_->schedule_output(
            U, precomputed_values, name, t, cycle, do_full_output, do_levelsets);
      }

      /* Checkpointing: */
      if (do_checkpointing) {
        Scope scope(computing_timer_, "time step [X] 4 - checkpointing");
        print_info("scheduling checkpointing");

        Checkpointing::write_checkpoint(
            *offline_data_, base_name_, U, t, cycle, mpi_communicator_);
      }
    }


    /*
     * Output and logging related functions:
     */


    template <typename Description, int dim, typename Number>
    void
    TimeLoopMgrit<Description, dim, Number>::print_parameters(std::ostream &stream)
    {

    }


    template <typename Description, int dim, typename Number>
    void
    TimeLoopMgrit<Description, dim, Number>::print_mpi_partition(std::ostream &stream)
    {

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::print_memory_statistics(
        std::ostream &stream)
    {

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::print_timers(std::ostream &stream)
    {

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::print_throughput(
        unsigned int cycle, Number t, std::ostream &stream, bool final_time)
    {

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::print_info(const std::string &header)
    {

    }


    template <typename Description, int dim, typename Number>
    void
    TimeLoopMgrit<Description, dim, Number>::print_head(const std::string &header,
        const std::string &secondary,
        std::ostream &stream)
    {

    }


    template <typename Description, int dim, typename Number>
    void TimeLoopMgrit<Description, dim, Number>::print_cycle_statistics(
        unsigned int cycle,
        Number t,
        unsigned int output_cycle,
        bool write_to_logfile,
        bool final_time)
    {

    }


  } // namespace mgrit
}//namespace ryujin
