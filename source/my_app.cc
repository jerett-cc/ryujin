#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <utility>

//ryujin includes
#include "checkpointing.h"
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "geometry_cylinder.h"
#include "discretization.h"
#include "hyperbolic_system.h"
#include "euler/parabolic_system.h"
#include "time_loop.h"
#include "euler/description.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"
#include "convenience_macros.h"
#include "local_index_handling.h"

//MPI
#include <deal.II/base/mpi.h>

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/tensor.h>

//xbraid include
#include <braid.h>
#include <braid.hpp>
#include <braid_test.h>

//mgrit includes
#include "my_app.h"
#include "mgrit_functions.template.h"

namespace mgrit{
  MyApp::MyApp(const MPI_Comm comm_x,
               const MPI_Comm comm_t,
               const std::vector<int> a_refinement_levels)
      : BraidApp(comm_t)
      , ParameterAcceptor("/App")
      , comm_x(comm_x)
      , levels(a_refinement_levels.size())
      , refinement_levels(a_refinement_levels)
      , time_loops(a_refinement_levels.size())
      , finest_level(0) // for XBRAID, the finest level is always 0.
  {
    coarsest_level = refinement_levels.size() - 1;
    print_solution_bool = false;
    add_parameter("print_solution_bool",
                  print_solution_bool,
                  "Optional to print the solution in Init and Access.");
    ntime = 10; // This is the default value in BraidApp(...)
    add_parameter("Time Bricks",
                  ntime,
                  "Number of time bricks total on the fine level.");
    tstart = 0.0;
    add_parameter("Start Time", tstart);
    tstop = 5.0;
    add_parameter("Stop Time", tstop);
    cfactor = 2; // The default coarsening from level to level is 2, which
                 // matches that of XBRAID.
    add_parameter(
        "cfactor", cfactor, "The coarsening factor between time levels.");
    max_iter = ntime; // In theory, mgrit should converge after the number of
                      // cycles equal to the number of time points it has.
    add_parameter(
        "max_iter", max_iter, "The maximum number of MGRIT iterations.");
    use_fmg = false;
    add_parameter(
        "use_fmg",
        use_fmg,
        "If set to true, this uses F-cycles."); // TODO: use this in main()
    n_relax = 1;
    add_parameter(
        "n relax",
        n_relax,
        "Number of relaxation steps: 1 is FC, 2 is FCF, 3 is FCFCF, etc.");
    access_level = 1; // This prints access only at the end of the simulation.
    add_parameter(
        "access_level",
        access_level,
        "The level of checking that access will do. 1 is at the end of "
        "the whole simulation, "
        "2 is every cycle, 3 is each interpolation and restriction and "
        "every function.");
  };

  MyApp::~MyApp(){};

  void MyApp::initialize(std::string prm_file)
  {
    // Reorder refinement levels in descending order of refinement,
    // this matches the fact that Xbraid has the finest level of MG
    // as 0. I.E. the most refined data is accessed with refinement_levels[0]
    std::sort(refinement_levels.rbegin(), refinement_levels.rend());
    // TODO: need to make a way to remove duplicates, or at least warn user
    // that duplicate refinement levels are inefficient.

    create_mg_levels();

    // now that levels are all created, we parse the parameter file.
    dealii::ParameterAcceptor::initialize(prm_file);

    // all parameters defined, we can now call all objects prepare function.
    prepare_mg_objects();
    //   initialized = true; // now the user can access data in app. TODO:
    //   implement a
    // check for getter functions.
  }

  void MyApp::create_mg_levels()
  {
    for (unsigned int i = 0; i < refinement_levels.size(); i++) {
      if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
        std::cout << "[INFO] Setting Structures in App at level "
                  << refinement_levels[i] << std::endl;
      }
      // TODO: determine if I should just make a time loop object for each level
      // and using only this.
      //  i.e. does app really ned to know all the level structures info?
      levels[i] = std::make_shared<
          ryujin::mgrit::LevelStructures<Description, 2, Number>>(
          comm_x, refinement_levels[i]);
      time_loops[i] =
          std::make_shared<ryujin::TimeLoop<Description, 2, Number>>(
              comm_x, *(levels[i]));
      std::cout << "Level " + std::to_string(refinement_levels[i]) + " created."
                << std::endl;
    }
  }

  void MyApp::prepare_mg_objects()
  {
    for (unsigned int lvl = 0; lvl < refinement_levels.size(); lvl++) {
      if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
        std::cout << "[INFO] Preparing Structures in App at level "
                  << refinement_levels[lvl] << std::endl;
      }
      levels[lvl]->prepare();
      std::cout << "Level " + std::to_string(refinement_levels[lvl]) +
                       " prepared."
                << std::endl;

      MPI_Barrier(MPI_COMM_WORLD); // TODO: need this?
    }
    // set the last variables in app.
    n_fine_dofs = levels[0]->offline_data->dof_handler().n_dofs();
    n_locally_owned_dofs = levels[0]->offline_data->n_locally_owned();
  }

  void MyApp::reinit_to_level(MyVector *u, const unsigned int level)
  {
    Assert(levels.size() > level,
           dealii::ExcMessage("The level being reinitialized does not exist."));
    u->U.reinit_with_scalar_partitioner(
        levels[level]->offline_data->scalar_partitioner());
    u->U.update_ghost_values(); // TODO: is this neccessary?
  }

  void MyApp::interpolate_between_levels(vector_type &to_v,
                                         const int to_level,
                                         const vector_type &from_v,
                                         const int from_level)
  {
    Assert((to_v.size() == levels[to_level]->offline_data->dof_handler().n_dofs()*problem_dimension)
      , dealii::ExcMessage("Trying to interpolate to a vector and level where the n_dofs do not match will not work."));
    Assert(
        (to_level != from_level),
        dealii::ExcMessage(
            "Levels you want to interpolate between are the same. to_level=" +
            std::to_string(to_level) +
            " from_level=" + std::to_string(from_level) +
            "this is a waste of time, just use the same vector, or make a "
            "copy."));

    using scalar_type = ryujin::OfflineData<2, NUMBER>::scalar_type;
    scalar_type from_component, to_component;

    std::cout << "Interpolating from level " << from_level << " to level "
              << to_level << std::endl;
    const auto &from_partitioner =
        levels[from_level]->offline_data->scalar_partitioner();
    const auto &from_dof_handler =
        levels[from_level]->offline_data->dof_handler();
    const auto &from_constraints =
        levels[from_level]->offline_data->affine_constraints();
    // const dealii::AffineConstraints<NUMBER> from_affine_constraints_tmp;
    // from_affine_constraints_tmp.copy_from(from_constraints);

    const auto &to_partitioner =
        levels[to_level]->offline_data->scalar_partitioner();
    const auto &to_dof_handler = levels[to_level]->offline_data->dof_handler();
    const auto &to_constraints =
        levels[to_level]->offline_data->affine_constraints();
    // const dealii::AffineConstraints to_affine_constraints_tmp;
    // from_affine_constraints_tmp.copy_from(to_constraints);

    // ryujin::transform_to_local_range(from_partitioner,
    //                                  from_affine_constraints_tmp);
    // ryujin::transform_to_local_range(to_partitioner, to_affine_constraints_tmp);

    const auto &comm = comm_x;

    // Reinit the components to match the correct info.
    from_component.reinit(from_partitioner, comm);
    from_constraints.distribute(from_component);

    to_component.reinit(to_partitioner, comm);
    to_constraints.distribute(
        to_component); // Do we need to do this for the vector to which we are
                       // interpolating?

    from_component.update_ghost_values();
    to_component.update_ghost_values(); // Do we need to do this for the vector
                                        // to which we are interpolating?

    // If the level we want to go to is less than the one we are from, we are
    // interpolating to a finer mesh, so we use the corresponding function.
    // Otherwise, we are interpolating to a coarser mesh, and use that function.

    if (from_level < to_level) {
      for (unsigned int comp = 0; comp < problem_dimension; comp++) {
        // extract component
        from_v.extract_component(from_component, comp);
        // interpolate this into the to_component
        dealii::VectorTools::interpolate_to_coarser_mesh(from_dof_handler,
                                                         from_component,
                                                         to_dof_handler,
                                                         to_constraints,
                                                         to_component);
        // place component
        to_v.insert_component(to_component, comp);
      }
    } else {
      for (unsigned int comp = 0; comp < problem_dimension; comp++) {
        // extract component
        from_v.extract_component(from_component, comp);
        // interpolate this into the to_component
        dealii::VectorTools::interpolate_to_finer_mesh(from_dof_handler,
                                                       from_component,
                                                       to_dof_handler,
                                                       to_constraints,
                                                       to_component);
        // place component
        to_v.insert_component(to_component, comp);
      }
    }
    to_v.update_ghost_values();
  }

  template <int dim>
  void MyApp::test_physicality(const vector_type u,
                               const int level,
                               std::string where)
  {
    std::cout << "Testing Physicality in location " + where << std::endl;
    const auto hs_view_level =
        levels[level]->hyperbolic_system->template view<dim, NUMBER>();

    // Check if the initial condition is admissible as a fluid state. It must
    // have positive density, entropy, and energy.
    for (unsigned int i = 0; i < levels[level]->offline_data->n_locally_owned();
         i++) {
      const bool is_admissible =
          hs_view_level.is_admissible(u.template get_tensor(i));
      Assert(is_admissible,
             dealii::ExcMessage("The state at index i=" + std::to_string(i) +
                                "is not admissible."));
      if (!is_admissible) {
        std::cout << "The state at index i=" + std::to_string(i) +
                         "is not admissible.\n"
                  << "State: " << u.template get_tensor(i) << std::endl;
      }

      const bool pressure_no_nans =
          (hs_view_level.pressure(u.template get_tensor(i)) ==
           hs_view_level.pressure(u.template get_tensor(i)));
      Assert(
          pressure_no_nans,
          dealii::ExcMessage("Pressure has a nan in one of the `lanes' at i= " +
                             std::to_string(i)));
      if (!pressure_no_nans) {
        std::cout << "Pressure is: "
                  << hs_view_level.pressure(u.template get_tensor(i))
                  << std::endl;
      }

      if (!pressure_no_nans || !is_admissible) {
        exit(EXIT_FAILURE);
      }
    }
  }

  void MyApp::print_solution(vector_type &v,
                             const double t,
                             const int level,
                             const std::string fname,
                             const bool time_in_fname,
                             const unsigned int cycle)
  {
    std::cout << "printing solution" << std::endl;
    const auto time_loop = time_loops[level];
    if (time_in_fname) {
      time_loop->output_wrapper(
          v, fname + std::to_string(t), t /*current time*/, 0 /*cycle*/);
    } else {
      time_loop->output_wrapper(v, fname, t /*current time*/, cycle /*cycle*/);
    }
  }

  unsigned int MyApp::n_locally_owned_at_level(const int level) const
  {
    return levels[level]->offline_data->n_locally_owned();
  }

  braid_Int MyApp::Step(braid_Vector u,
                        braid_Vector ustop,
                        braid_Vector fstop,
                        BraidStepStatus &pstatus)
  {
    MyVector *u_ = (MyVector*) u;
    // this variable is used for writing data to
    // different files during the parallel computations.
    // is passed to run_with_initial_data
    static unsigned int num_step_calls = 0;

    // grab the MG level for this step
    int level;
    pstatus.GetLevel(&level);
    // grab the start time and end time
    double lvl_tstart;
    double lvl_tstop;
    pstatus.GetTstartTstop(&lvl_tstart, &lvl_tstop);

    // Ensure this is a physical vector.
    mgrit_functions::
        enforce_physicality_bounds<Description, 2, NUMBER>(
            *u_, finest_level, *this, lvl_tstart);

    if (1 /*this was a mpi comm id check before FIXME*/) {
      std::cout << "[INFO] Stepping on level: " + std::to_string(level) +
                       "\non interval: [" + std::to_string(lvl_tstart) + ", " +
                       std::to_string(lvl_tstop) + "]\n" +
                       "total step call number " +
                       std::to_string(num_step_calls)
                << std::endl;
    }

    std::string fname = "step" + std::to_string(num_step_calls) + "_cycle" +
                        std::to_string(n_cycles) + "_level_" +
                        std::to_string(level) + "_interval_[" +
                        std::to_string(lvl_tstart) + "_" + std::to_string(lvl_tstop) +
                        "]";
#ifdef CHECK_BOUNDS
    // Test that the incoming vector is physical at the fine level.
    test_physicality<2>(u_->U, 0, "before interpolation.");
#endif

    // use a macro to get rid of some unused variables to avoid -Wall messages
    UNUSED(ustop);
    UNUSED(fstop);

    // translate the fine level u coming in to the coarse level
    // this uses a function from DEALII interpolate to different mesh

    // new, coarse vector
    MyVector *u_to_step = new (MyVector);
    reinit_to_level(u_to_step, level);

    std::cout << "norm of initialized u_to_step: " << u_to_step->U.l2_norm() << std::endl;

    // interpolainterpolate_between_levels(te between levels, put data from u (fine level) onto the
    // u_to_step (coarse level)
    interpolate_between_levels(u_to_step->U, level, u_->U, 0);

#ifdef CHECK_BOUNDS
    // Test physicality of interpolated vector.
    test_physicality<2>(u_to_step->U, level, "before step.");
#endif

    if (print_solution_bool)
      print_solution(u_to_step->U,
                     lvl_tstart,
                     level,
                     fname,
                     false,
                     n_cycles);

    if (level == 1 && std::abs(lvl_tstart - 3.125) < 1e-6 &&
        std::abs(lvl_tstop - 3.4375) < 1e-6 && num_step_calls == 132)
      ryujin::Checkpointing::write_checkpoint(
          *(levels[level]->offline_data),
          fname,
          u_to_step->U,
          lvl_tstart,
          num_step_calls,
          comm_x);
    // step the function on this level
    time_loops[level]->change_base_name(fname);
    time_loops[level]->run_with_initial_data(
        u_to_step->U,
        lvl_tstop,
        lvl_tstart,
        false /*print every step of this simulation*/);

#ifdef CHECK_BOUNDS
    // Test physicality of vector after it has been stepped.
    test_physicality<2>(u_to_step->U, level, "after step.");
#endif

    double norm = u_to_step->U.l1_norm();

    if (!dealii::numbers::is_finite(norm)) {
      // ryujin::Checkpointing::write_checkpoint(
      //         *(levels[level]->offline_data), fname, u_to_step->U, lvl_tstop,
      //         num_step_calls, comm_x);
      std::cout << "nan in file " << fname << std::endl;
      std::cout << "Norm was " << norm << std::endl;
      // exit(EXIT_FAILURE);
    }
    // interpolate this back to the fine level
    interpolate_between_levels(u_->U, 0, u_to_step->U, level);

#ifdef CHECK_BOUNDS
    // Test physicality of interpolated vector on fine level, after the step.
    test_physicality<2>(
        u_->U, 0, "after step, after interpolation.");
#endif

    std::string fname_post = "FcOnLevel_" + std::to_string(level) +
                             "on_interval_[" + std::to_string(lvl_tstart) + "_" +
                             std::to_string(lvl_tstop) + "]";
    if (print_solution_bool)
      print_solution(u_->U,
                     lvl_tstop,
                     0 /*level, always needs to be zero, to be fixed*/,
                     fname_post,
                     false,
                     n_cycles);

    num_step_calls++;
    // done.

    delete u_to_step;
    return 0;
  }

  braid_Int
  MyApp::Residual(braid_Vector u, braid_Vector r, BraidStepStatus &pstatus)
  {
    /// Does nothing.
    UNUSED(u);
    UNUSED(r);
    UNUSED(pstatus);
    return 0;
  }

  braid_Int MyApp::Clone(braid_Vector u, braid_Vector *v_ptr)
  {
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Cloning XBraid vectors" << std::endl;
    }
    MyVector *u_ = (MyVector *) u;
    MyVector *v = new (MyVector);
    // all vectors are 'fine level' vectors
    reinit_to_level(v, 0);
    v->U.equ(1, u_->U);

    *v_ptr = (braid_Vector)v;

    return 0;
  }

  braid_Int MyApp::Init(braid_Real t, braid_Vector *u_ptr)
  {
    std::cout << "[INFO] Initializing XBraid vectors at t=" << t << std::endl;

    // We first define a coarse vector, at the coarsest level, which will be
    // stepped, then restricted down to the fine level and interpolate the fine
    // initial state into the coarse vector, then interpolates it up to the
    // coarse level and steps.
    MyVector *u = new (MyVector);
    MyVector *temp_coarse = new (MyVector);
    reinit_to_level(
        u,
        finest_level); // this is the u that we will start each time brick with.
    reinit_to_level(temp_coarse, coarsest_level); // coarse on the coarses
                                                  // level.
    // sets up U data at t=0;
    u->U = levels[finest_level]->initial_values->interpolate(0); 
    temp_coarse->U = levels[coarsest_level]->initial_values->interpolate(0);

    std::string str = "initialized_at_t=" + std::to_string(t);
    // If T is not zero, we step on the coarsest level until we are done.
    // Otherwise we have no need to step any because the assumtion is that T=0
    // TODO: implicit assumption that T>0 always here except for T=0.
    if (std::fabs(t) > 0) {
      // interpolate the initial conditions up to the coarsest mesh
      interpolate_between_levels(
          temp_coarse->U, coarsest_level, u->U, finest_level);
      // steps to the correct end time on the coarse level to end time t
      time_loops[coarsest_level]->run_with_initial_data(temp_coarse->U, t);
      if (print_solution_bool)
        print_solution(temp_coarse->U, t, coarsest_level, str);

      interpolate_between_levels(
          u->U, finest_level, temp_coarse->U, coarsest_level);
    }
    if (print_solution_bool)
      print_solution(u->U,
                     t,
                     finest_level,
                     str,
                     false,
                     -1); // prints the interpolated state.
    // delete the temporary coarse U. f
    delete temp_coarse;

    // FIXME: the whole cpp interface as awkward use of pointers for the vector objects.
    if( !(u->U.l1_norm()) ){
      std::cout << "Norm of u_ptr is not one." << std::endl;
      exit(EXIT_FAILURE);
    }

    // reassign pointer XBraid will use
    *u_ptr = (braid_Vector)u;
    return 0;
  }

  braid_Int MyApp::Free(braid_Vector u)
  {
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Freeing XBraid vectors" << std::endl;
    }
    MyVector *u_ = (MyVector*) u;
    delete u_;

    return 0;
  }

  braid_Int MyApp::Sum(braid_Real alpha,
                       braid_Vector x,
                       braid_Real beta,
                       braid_Vector y)
  {
    // keep track of the number of times this has been called
    static int sum_count = 0;
    MyVector *x_ = (MyVector *) x;
    MyVector *y_ = (MyVector *) y;

    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Summing XBraid vectors" << std::endl;
      std::cout << alpha << "x + " << beta << "y" << std::endl;
    }
// #ifdef CHECK_BOUNDS
//     if (0 < alpha && 0 < beta)
//       test_physicality<2>(y_->U, 0, "my_Sum: incoming y");
// #endif

    y_->U.sadd(beta, alpha, x_->U);
// #ifdef CHECK_BOUNDS
//     // Test that the outgoing vector is physical at the fine level, but only for
//     // the prolongation step.
//     if (0 < alpha && 0 < beta)
//       test_physicality<2>(y_->U, 0, "my_Sum: y after summing.");
// #endif

    sum_count++;

    return 0;
  }

  braid_Int MyApp::SpatialNorm(braid_Vector u, braid_Real *norm_ptr)
  {
    MyVector *u_ = (MyVector *) u;
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Calculating XBraid vector spatial norm" << std::endl;
    }

    *norm_ptr = u_->U.l2_norm();

    return 0;
  }

  braid_Int MyApp::Access(braid_Vector u, BraidAccessStatus &astatus)
  {
    MyVector *u_ = (MyVector *) u;

    braid_Int caller_id;
    static int mgCycle = 0;
    double t = 0;
    braid_Int t_idx;

    // state what iteration we are on, and what time t we are at.
    astatus.GetCallingFunction(&caller_id);
    astatus.GetIter(&mgCycle);
    astatus.GetT(&t);
    astatus.GetTIndex(&t_idx);

   std::string fname = "./cycle" + std::to_string(mgCycle);

    switch (caller_id) 
    {
      case braid_ASCaller_FInterp_Projection:
      {
        fname = fname + "caller_FInterp_Projection";
        std::cout << "Access called for " + fname
                  << " enforcing physicality bounds after summing in FInterp."
                  << std::endl;
        // Call the stability projection function.
        mgrit_functions::enforce_physicality_bounds<Description, 2, Number>(
            *u_, finest_level, *this, t);
#ifdef CHECK_BOUNDS
        test_physicality<2>(
            u_->U, 0, "my_Access: u when caller is FInterp_Projection.");
#endif
        break;
      }
      case braid_ASCaller_FAccess: 
      {
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
          std::cout << "[INFO] Access Called" << std::endl;
        }

        if (print_solution_bool &&
            (std::abs(t - 0) < 1e-6 || std::abs(t - 1.25) < 1e-6 ||
             std::abs(t - 2.5) < 1e-6 || std::abs(t - 3.75) < 1e-6 ||
             std::abs(t - 5.0) <
                 1e-6)) { // FIXME: this only prints for the [0,5] time interval
                          // at specific points. Make this more general.
          print_solution(
              u_->U, t, finest_level /*level that u lives on*/, fname, false,t_idx);
        }
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
          std::cout << "Cycles done: " << mgCycle << std::endl;
        }
        // calculate drag (at end of cycle...)
        dealii::Tensor<1, 2> forces =
            mgrit_functions::calculate_drag_and_lift<2>(this, *u_, t);
        std::cout << "cycle." + std::to_string(mgCycle) + " drag." +
                         std::to_string(forces[0]) + " lift." +
                         std::to_string(forces[1]) + " time." +
                         std::to_string(t)
                  << std::endl;

        n_cycles = mgCycle;
        break;
      }
      default:
      {
        break;
      }
    }

      return 0;
  }

  braid_Int MyApp::BufSize(braid_Int *size_ptr,
                           BraidBufferStatus &bstatus)
  {
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Buf_size Called" << std::endl;
    }

    // TODO: answer question about what the buffer size whould be, i think it
    // should be problem_dimension*number of spatial nodes. But there is some
    // question in my mind about if the MPI communication is from this Time
    // Brick (which owns a distributed vector on a few processors) or among the
    // spatial processors. I suspect the former.
    UNUSED(bstatus);

    // no vector can be bigger than this, so we are very conservative.
    int size = n_fine_dofs * problem_dimension;
    *size_ptr =
        (size + 1) * sizeof(NUMBER); //+1 is for the size of the buffers being
                                     // stored in the first component.
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "Size in bytes of the NUMBER: " << sizeof(NUMBER)
                << std::endl;
      std::cout << "Problem_dimension: " << problem_dimension
                << " n_dofs: " << n_fine_dofs << std::endl;
      std::cout << "buf_size: " << *size_ptr << std::endl;
    }

    return 0;
  }

  braid_Int MyApp::BufPack(braid_Vector u,
                           void *buffer,
                           BraidBufferStatus &bstatus)
  {
    MyVector *u_ = (MyVector *) u;
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] BufPack Called" << std::endl;
    }

    NUMBER *dbuffer = (NUMBER *)buffer;
    unsigned int n_locally_owned =
        n_locally_owned_dofs; // number of dofs at finest level
    unsigned int buf_size = n_locally_owned * problem_dimension;
    dbuffer[0] = buf_size + 1; // buffer + size
    for (unsigned int node = 0; node < n_locally_owned; node++) {
      for (unsigned int component = 0; component < problem_dimension;
           ++component) {
        Assert(
            buf_size >= (node + component),
            dealii::ExcMessage("In my_BufPack, the size of node + component is "
                               "greater than the buff_size (the expected size "
                               "of the vector)."));
        dbuffer[problem_dimension * (node) + component + 1] =
            u_->U.local_element(problem_dimension * node + component);
        Assert(!std::isnan(
                   u_->U.local_element(problem_dimension * node + component)),
               dealii::ExcMessage(
                   "The vector you are trying to pack has a NaN in it at "
                   "component " +
                   std::to_string(problem_dimension * node + component)));
      }
    }
    bstatus.SetSize(
        (buf_size + 1) *
        sizeof(NUMBER)); // set the number of bytes stored in this buffer (TODO:
                         // this is off since the dbuffer[0] is a integer.)
    return 0;
  }

  braid_Int MyApp::BufUnpack(void *buffer,
                             braid_Vector *u_ptr,
                             BraidBufferStatus &bstatus)
  {
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] BufUnpack Called" << std::endl;
    }

    UNUSED(bstatus);

    NUMBER *dbuffer = (NUMBER *)buffer;
    // todo: use this for a range check. Make sure we are not indexing outside of bounds of the buffer.
    // unsigned int buf_size = static_cast<unsigned int>(dbuffer[0]); // TODO: is this dangerous?

    // The vector should be size (dim + 2) X n_dofs at finest level.
    MyVector *u = new (MyVector); // TODO: where does this get deleted? Probably
                                  // wherever owns the u_ptr.
    reinit_to_level(u, finest_level); // each U is at the finest level.

    // unpack the sent data into the right level
    for (unsigned int node = 0; node < n_locally_owned_dofs; node++) {
      // get tensor at node.
      for (unsigned int component = 0; component < problem_dimension;
           ++component) {
        u->U.local_element(problem_dimension * node + component) =
            dbuffer[problem_dimension * node + component +
                    1]; // TODO: test for speed.
        Assert((problem_dimension * node + component + 1 <= buf_size),
               dealii::ExcMessage(
                   "Somehow, you are exceeding the buffer size as you unpack."
                   " here, buf_size is " +
                   std::to_string(buf_size) +
                   ", and the place you are trying to access is " +
                   std::to_string(problem_dimension * node + component + 1)));
      }
    }

    *(u_ptr) = (braid_Vector)u; // modify the u_ptr does this create a memory leak as we just
                  // point this pointer somewhere else?

#ifdef CHECK_BOUNDS
    // Test that the outgoing vector is physical at the fine level.
    test_physicality<2>(u->U, 0, "my_BufUnpack: unpacked vector.");
#endif

    return 0;
  }
}//Namepsace mgrit
