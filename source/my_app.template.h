#pragma once

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
#include "time_integrator.template.h"
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

  template<typename Number, typename Description, int dim>
  MyApp<Number, Description, dim>::MyApp(const MPI_Comm comm_x,
               const MPI_Comm comm_t,
               const std::vector<int> a_refinement_levels)
      : BraidApp(comm_t)
      , ParameterAcceptor("/App")
      , comm_x(comm_x)
      , levels(a_refinement_levels.size())
      , refinement_levels(a_refinement_levels)
      , time_loops(a_refinement_levels.size())
      , finest_level(0) // for XBRAID, the finest level is always 0.
      , discretization_vec(1)
      , offline_data_vec(1) // initialize this with only one level, will resize later.
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

  template<typename Number, typename Description, int dim>
  MyApp<Number, Description, dim>::~MyApp(){};

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::initialize(std::string prm_file)
  {
    ryujin::Scope scope(computing_timer, "initialize");
    // Reorder refinement levels in descending order of refinement,
    // this matches the fact that Xbraid has the finest level of MG
    // as 0. I.E. the most refined data is accessed with refinement_levels[0]
    std::sort(refinement_levels.rbegin(), refinement_levels.rend());
    // Test that the refinement levels are in the right order.
    Assert(
        (refinement_levels.front() - refinement_levels.back() >= 0),
        dealii::ExcMessage(
            "Refinement levels is not ordered in a proper way. Here, front()=" +
            std::to_string(refinement_levels.front()) +
            " and back()=" + std::to_string(refinement_levels.back())));
        // TODO: need to make a way to remove duplicates, or at least warn user
        // that duplicate refinement levels are inefficient.

    create_mg_levels();

    // Set up the offline_data_vec a vector of pointers to all the
    // offline_data's for all levels between the finest and coarsest levels we
    // actually care about.
    const int n_total_refinements =
        refinement_levels.front() - refinement_levels.back()+1;//inclusive
    const int most_refinement = refinement_levels.front();//index of most refined obj
    const int least_refinement = refinement_levels.back();//index of least refined obj

    discretization_vec.resize(n_total_refinements);
    offline_data_vec.resize(n_total_refinements);
    level_map[0] = 0;//The finest level is always at index 0.

    int iter = 0;
    for (int lvl = most_refinement; 
         lvl >= least_refinement;
         lvl--) {
      std::cout << lvl << std::endl;
      if (std::find(refinement_levels.begin(),
                    refinement_levels.end(),
                    lvl) != refinement_levels.end()) {
                      std::cout << "exists"<< std::endl;
        discretization_vec[most_refinement-lvl] = levels[iter]->discretization;
        offline_data_vec[most_refinement-lvl] = levels[iter]->offline_data;
        level_map[iter] =
            most_refinement -
            lvl; // The lvl, if it is one we care about for computations, lives
                 // in the offline_data_vec at index most_refinement-lvl, which
                 // in principle is not the same at lvl.
        iter++;
      } else {
        std::cout << "does not exist" << std::endl;
        discretization_vec[most_refinement-lvl] = std::make_shared<DiscretizationType>(
            comm_x, lvl, "/C - Discretization");
        offline_data_vec[most_refinement-lvl] = std::make_shared<OfflineDataType>(comm_x, 
                                                                *discretization_vec[most_refinement-lvl],
                                                                "/OfflineData");
      }
    }
    // now that levels are all created, we parse the parameter file.
    dealii::ParameterAcceptor::initialize(prm_file);

    // all parameters defined, we can now call all objects prepare function.
    prepare_mg_objects();

    // Prepare the additional offline_data and discretizations.
    for (int lvl = most_refinement; lvl >= least_refinement; lvl--) {
      // If we don't find this level already set up, we prepare it.
      if (std::find(refinement_levels.begin(),
                    refinement_levels.end(),
                    lvl) == refinement_levels.end()) {
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
          std::cout << "Preparing additional offline_data and "
                       "discretization for interpolation purposes."
                    << std::endl;
        }
        discretization_vec[most_refinement-lvl]->prepare();
        offline_data_vec[most_refinement-lvl]->prepare(problem_dimension, false);
      }
    }

    //   initialized = true; // now the user can access data in app. TODO:
    //   implement a
    // check for getter functions.
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::create_mg_levels()
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
          ryujin::mgrit::LevelStructures<Description, dim, Number>>(
          comm_x, refinement_levels[i]);
      time_loops[i] =
          std::make_shared<ryujin::TimeLoop<Description, dim, Number>>(
              comm_x, *(levels[i]));
      std::cout << "Level " + std::to_string(refinement_levels[i]) + " created."
                << std::endl;
    }
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::prepare_mg_objects()
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

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::reinit_to_level(my_vector *u, const int level)
  {
    Assert(levels.size() > static_cast<unsigned int>(level),
           dealii::ExcMessage("The level being reinitialized does not exist."));
    std::get<0>(u->U).reinit(levels[level]->offline_data->hyperbolic_vector_partitioner());
    std::get<1>(u->U).reinit(levels[level]->offline_data->precomputed_vector_partitioner());
    std::get<0>(u->U).update_ghost_values(); // TODO: is this neccessary?
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::interpolate_between_levels(vector_type &to_v,
                                         const int to_level,
                                         const vector_type &from_v,
                                         const int from_level)
  {
    Assert(
        (to_v.size() == levels[to_level]->offline_data->dof_handler().n_dofs() *
                            problem_dimension),
        dealii::ExcMessage("Trying to interpolate to a vector and level where "
                           "the n_dofs do not match will not work."));
    Assert(((to_level >= 0) && (from_level >= 0)),
           dealii::ExcMessage("You cannot interpolate to or from a level that "
                              "is negaitve, all levels are non-negative."));

    ryujin::Scope scope(computing_timer, "interpolate_between_levels");

    // If both levels are equal, we simply copy the data from_vector and put it in to_vector.
    // Otherwise, we actually need to do some computation.
    if(to_level == from_level)
    {
      // Copy the data using the dealii::operator= for distributed vectors, and
      // nothing else.
      // TODO: does this do what I think, leaving the from_v alone? Is it better
      // to have the else case wrapped in an else statement? Ask Wolfgang.
      to_v = from_v;
      return;
    }

    scalar_type next_component, curr_component;

    // First, set up a vector of pointers to vectors which will correspond to
    // data at each level, inclusive of th e level we start interpolation.
    std::vector<vector_type*> level_vectors(std::abs(level_map[from_level]-level_map[to_level])+1);
    
    // Initialize each of these TODO: memory unsafe? see end of function.
    for(auto &lvl_v : level_vectors)
      lvl_v = new vector_type();
    delete level_vectors[0];//remove the first one since we immediately replace it with a temp.
    // Copy the incoming data to be interpolated.
    vector_type* CV = new vector_type(from_v);
    // Store this as the first entry in the temporary vector.
    level_vectors[0] = CV;
    
    const bool up = true;
    const bool down = false;
    // Figure out the direction we need to loop, up or down. Set start_lvl and end_lvl accordingly
    const bool dir = (to_level < from_level) ? down : up;

    // Looping from start to end, interpolate from curr_lvl to curr_lvl +- 1 and interpolate, 
    // until we reach the stop_lvl. Once we reach the last lvl, set next_v to be the to_v
    int lvl_iter =0;
    if(dir == up)//corresponts to ++ and incrementing with +
    {
      for(int curr_lvl = level_map[from_level]; curr_lvl < level_map[to_level]; curr_lvl++)
      {
        const int next_lvl = curr_lvl+1;
        Assert(((unsigned int)(lvl_iter+1) < level_vectors.size()),
               dealii::ExcMessage("The next level in the interpolation will "
                                  "index you out of bounds."));
        vector_type* curr_v = level_vectors[lvl_iter];
        // If the next level is out last, we will be modifying the to_v, not one
        // of the temp_vectors.
        vector_type* next_v = (next_lvl != level_map[to_level]) ?  level_vectors[lvl_iter+1] : &to_v;
        const auto &curr_od = offline_data_vec[curr_lvl];
        const auto &curr_dof_handl = curr_od->dof_handler();

        const auto &next_od = offline_data_vec[next_lvl];
        const auto &next_dof_handl = next_od->dof_handler();
        const auto &next_constraints = next_od->affine_constraints();
        
        // If we are not on the final level, we will need to reinit the temp vector.
        if(next_lvl != level_map[to_level])
          next_v->reinit_with_scalar_partitioner(next_od->scalar_partitioner());
        curr_component.reinit(curr_od->scalar_partitioner(), comm_x);
        next_component.reinit(next_od->scalar_partitioner(), comm_x);

        Assert(
            (curr_dof_handl.get_triangulation().n_levels() ==
             next_dof_handl.get_triangulation().n_levels() + 1),
            dealii::ExcMessage(
                "For interpolation, you can only interpolate between two "
                "levels whos difference in levels is 1, which corresponds "
                "to only one level of refinement that differentiates them. "
                "Here, the coarser mesh has n_levels=" +
                std::to_string(next_dof_handl.get_triangulation().n_levels()) +
                " and the finer mesh has n_levels=" +
                std::to_string(curr_dof_handl.get_triangulation().n_levels())));
        // Extract and interpolate components.
        for (unsigned int c = 0; c < problem_dimension; c++) 
        {
          // Extract comonent from curr_v
          curr_v->extract_component(curr_component, c);
          // A scope here to independently time the interpolation function.
          {
          ryujin::Scope scope(computing_timer, "interpolate_to_coarser_mesh");
          // Up also means we are interpolating to a coarser mesh.
          dealii::VectorTools::interpolate_to_coarser_mesh(curr_dof_handl,
                                                           curr_component,
                                                           next_dof_handl,
                                                           next_constraints,
                                                           next_component);
          }
          // Place component in next_v
          next_v->insert_component(next_component,c);
        }
        lvl_iter++;
      }
    } else if (dir == down) {
      // Down means we decrement the level.
      Assert((level_map[from_level] > level_map[to_level]),
             dealii::ExcMessage(
                 "When interpolating to a down to a finer mesh, the index in "
                 "the total "
                 "levels vector of the from_level=" +
                 std::to_string(from_level) + " which maps to index" +
                 std::to_string(level_map[from_level]) +
                 " needs to be bigger than the to_level=" +
                 std::to_string(to_level) + " which maps to index " +
                 std::to_string(level_map[to_level])));

      // Decrement through the levels.
      for(int curr_lvl = level_map[from_level]; curr_lvl > level_map[to_level]; curr_lvl--)
      {
        const int next_lvl = curr_lvl - 1;
        Assert(
            ((unsigned int)lvl_iter < level_vectors.size() && next_lvl >= level_map[to_level]),
             dealii::ExcMessage(
                 "The next level in the interpolation will "
                 "index you out of bounds, below zero, or the lvl_iter is "
                 "larger than the level_vectors.size()"));
        vector_type *curr_v = level_vectors[lvl_iter];
        // If the next level is out last, we will be modifying the to_v, not
        // one of the temp_vectors.
        vector_type *next_v =
            (next_lvl != level_map[to_level]) ? level_vectors[lvl_iter + 1] : &to_v;
        const auto &curr_od = offline_data_vec[curr_lvl];
        const auto &curr_dof_handl = curr_od->dof_handler();

        const auto &next_od = offline_data_vec[next_lvl];
        const auto &next_dof_handl = next_od->dof_handler();
        const auto &next_constraints = next_od->affine_constraints();

        // If we are not on the final level, we will need to reinit the temp
        // vector.
        if (next_lvl != level_map[to_level])
          next_v->reinit_with_scalar_partitioner(next_od->scalar_partitioner());
        curr_component.reinit(curr_od->scalar_partitioner(), comm_x);
        next_component.reinit(next_od->scalar_partitioner(), comm_x);

        // TODO: this assert is large, probably unnessesarily, refactor?
        // Check that we actually are interpolating between two levels who
        // differ only by one level.
        Assert((curr_dof_handl.get_triangulation().n_levels() ==
                next_dof_handl.get_triangulation().n_levels() + 1) || 
                (curr_dof_handl.get_triangulation().n_levels() + 1 ==
                next_dof_handl.get_triangulation().n_levels()),
               dealii::ExcMessage(
                   "For interpolation, you can only interpolate between two "
                   "levels whos difference in levels is 1, which corresponds "
                   "to only one level of refinement that differentiates them. "
                   "Here, the coarser mesh has n_levels=" +
                   std::to_string(
                       next_dof_handl.get_triangulation().n_levels()) +
                   " and the finer mesh has n_levels=" +
                   std::to_string(
                       curr_dof_handl.get_triangulation().n_levels())));
        // Extract and interpolate components.
        for (unsigned int c = 0; c < problem_dimension; c++) 
        {
          // Extract comonent from curr_v
          curr_v->extract_component(curr_component, c);
          {
          ryujin::Scope scope(computing_timer, "interpolate_to_finer_mesh");
          // Down means we are interpolating to a finer mesh.
          dealii::VectorTools::interpolate_to_finer_mesh(curr_dof_handl,
                                                         curr_component,
                                                         next_dof_handl,
                                                         next_constraints,
                                                         next_component);
          }
          // Place component in next_v
          next_v->insert_component(next_component,c);
        }
        lvl_iter++;
      }

      // Delete all temp vectors.
      // TODO: This seems like poor practice, but for some reason,
      // std::unique/shared_ptr is not working. Thread safety issue? I feel like
      // this might accidentally delete the to/from vectors if you  are not careful.
      for (auto &lvl_v : level_vectors)
        delete lvl_v;
    }
    to_v.update_ghost_values();
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::test_physicality(const vector_type u,
                               const int level,
                               std::string where)
  {
    ryujin::Scope scope(computing_timer, "test_physicality");
    std::cout << "Testing Physicality in location " + where << std::endl;
    const auto hs_view_level =
        levels[level]->hyperbolic_system->template view<dim, Number>();

    // Check if the initial condition is admissible as a fluid state. It must
    // have positive density, entropy, and energy.
    for (unsigned int i = 0; i < levels[level]->offline_data->n_locally_owned();
         i++) {
      const bool is_admissible =
          hs_view_level.is_admissible(u.template get_tensor(i));
#ifdef DEBUG
      if (!is_admissible) {
        std::cout << "The state at index i=" + std::to_string(i) +
                         "is not admissible.\n"
                  << "State: " << u.template get_tensor(i) << std::endl;
      }
  #endif
      Assert(is_admissible,
             dealii::ExcMessage("The state at index i=" + std::to_string(i) +
                                "is not admissible."));

      const bool pressure_no_nans =
          (hs_view_level.pressure(u.template get_tensor(i)) ==
           hs_view_level.pressure(u.template get_tensor(i)));
#ifdef DEBUG
      if (!pressure_no_nans) {
        std::cout << "Pressure is: "
                  << hs_view_level.pressure(u.template get_tensor(i))
                  << std::endl;
      }
#endif
      Assert(
          pressure_no_nans,
          dealii::ExcMessage("Pressure has a nan in one of the `lanes' at i= " +
                             std::to_string(i)));

      if (!pressure_no_nans || !is_admissible) {
        exit(EXIT_FAILURE);
      }
    }
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::print_solution(StateVector &v,
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

  template<typename Number, typename Description, int dim>
  unsigned int MyApp<Number, Description, dim>::n_locally_owned_at_level(const int level) const
  {
    return levels[level]->offline_data->n_locally_owned();
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Step(braid_Vector u,
                        braid_Vector ustop,
                        braid_Vector fstop,
                        BraidStepStatus &pstatus)
  {
    my_vector *u_ = (my_vector*) u;
    // this variable is used for writing data to
    // different files during the parallel computations.
    // is passed to run_with_initial_data
    static unsigned int num_step_calls = 0;

    // grab the MG level for this step
    int level;
    pstatus.GetLevel(&level);
    // Start a timer for step::level
    ryujin::Scope scope(computing_timer, "step::" + std::to_string(level));

    // grab the start time and end time
    double lvl_tstart;
    double lvl_tstop;
    pstatus.GetTstartTstop(&lvl_tstart, &lvl_tstop);

    // Ensure this is a physical vector.
    mgrit_functions::
        enforce_physicality_bounds<Description, dim, Number>(
            *u_, finest_level, *this, lvl_tstart);

#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Stepping on level: " + std::to_string(level) +
                       "\non interval: [" + std::to_string(lvl_tstart) + ", " +
                       std::to_string(lvl_tstop) + "]\n" +
                       "total step call number " +
                       std::to_string(num_step_calls)
                << std::endl;
    }
#endif

    std::string fname = "step" + std::to_string(num_step_calls) + "_cycle" +
                        std::to_string(n_cycles) + "_level_" +
                        std::to_string(level) + "_interval_[" +
                        std::to_string(lvl_tstart) + "_" + std::to_string(lvl_tstop) +
                        "]";
#ifdef CHECK_BOUNDS
    // Test that the incoming vector is physical at the fine level.
    test_physicality(std::get<0>(u_->U), 0, "before interpolation.");
#endif

    // use a macro to get rid of some unused variables to avoid -Wall messages
    UNUSED(ustop);
    UNUSED(fstop);

    // translate the fine level u coming in to the coarse level
    // this uses a function from DEALII interpolate to different mesh

    // new, coarse vector if levels are not the same, or, a copy of the fine
    // vector, if levels are both finest_level.
    my_vector *u_to_step = new (my_vector);
    reinit_to_level(u_to_step, level);

    std::cout << "norm of initialized u_to_step: " << std::get<0>(u_to_step->U).l2_norm() << std::endl;

    // Interpolate between levels, put data from u (fine level) onto the
    // u_to_step (coarse level), if the level is not zero (this is because all
    // the vectors are assumed to be at the finest level spatially.) This allows
    // computations which are naturally faster on the coarser levels, due to a
    // larger mesh size.

    interpolate_between_levels(std::get<0>(u_to_step->U), level, std::get<0>(u_->U), 0);

#ifdef CHECK_BOUNDS
    // Test physicality of interpolated vector.
    test_physicality(std::get<0>(u_to_step->U), level, "before step.");
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
          std::get<0>(u_to_step->U),
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
    test_physicality(std::get<0>(u_to_step->U), level, "after step.");
#endif

    double norm = std::get<0>(u_to_step->U).l1_norm();

    if (!dealii::numbers::is_finite(norm)) {
      // ryujin::Checkpointing::write_checkpoint(
      //         *(levels[level]->offline_data), fname, u_to_step->U, lvl_tstop,
      //         num_step_calls, comm_x);
      std::cout << "nan in file " << fname << std::endl;
      std::cout << "Norm was " << norm << std::endl;
      // exit(EXIT_FAILURE);
    }
    // Interpolate the updated state back to the fine level.
    interpolate_between_levels(std::get<0>(u_->U), 0, std::get<0>(u_to_step->U), level);

#ifdef CHECK_BOUNDS
    // Test physicality of interpolated vector on fine level, after the step.
    test_physicality(
        std::get<0>(u_->U), 0, "after step, after interpolation.");
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

  template<typename Number, typename Description, int dim>
  braid_Int
  MyApp<Number, Description, dim>::Residual(braid_Vector u, braid_Vector r, BraidStepStatus &pstatus)
  {
    /// Does nothing.
    UNUSED(u);
    UNUSED(r);
    UNUSED(pstatus);
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Clone(braid_Vector u, braid_Vector *v_ptr)
  {
#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Cloning XBraid vectors" << std::endl;
    }
#endif
    ryujin::Scope scope(computing_timer, "clone");
    my_vector *u_ = (my_vector *) u;
    my_vector *v = new (my_vector);
    // all vectors are 'fine level' vectors
    reinit_to_level(v, 0);
    std::get<0>(v->U).equ(1, std::get<0>(u_->U));

    *v_ptr = (braid_Vector)v;

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Init(braid_Real t, braid_Vector *u_ptr)
  {
    std::cout << "[INFO] Initializing XBraid vectors at t=" << t << std::endl;

    // We first define a coarse vector, at the coarsest level, which will be
    // stepped, then restricted down to the fine level and interpolate the fine
    // initial state into the coarse vector, then interpolates it up to the
    // coarse level and steps.
    my_vector *u = new (my_vector);
    my_vector *temp_coarse = new (my_vector);
    reinit_to_level(
        u,
        finest_level); // this is the u that we will start each time brick with.
    reinit_to_level(temp_coarse, coarsest_level); // coarse on the coarses
                                                  // level.
    // sets up U data at t=0;
    u->U = levels[finest_level]->initial_values->interpolate_state_vector(0); 
    temp_coarse->U = levels[coarsest_level]->initial_values->interpolate_state_vector(0);

    std::string str = "initialized_at_t=" + std::to_string(t);
    // If T is not zero, we step on the coarsest level until we are done.
    // Otherwise we have no need to step any because the assumtion is that T=0
    // TODO: implicit assumption that T>0 always here except for T=0.
    if (std::fabs(t) > 0) {
      // interpolate the initial conditions up to the coarsest mesh
      interpolate_between_levels(
          std::get<0>(temp_coarse->U), coarsest_level, std::get<0>(u->U), finest_level);
      // steps to the correct end time on the coarse level to end time t
      time_loops[coarsest_level]->run_with_initial_data(temp_coarse->U, t);
      if (print_solution_bool)
        print_solution(temp_coarse->U, t, coarsest_level, str);

      interpolate_between_levels(
          std::get<0>(u->U), finest_level, std::get<0>(temp_coarse->U), coarsest_level);
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
    if( !(std::get<0>(u->U).l1_norm()) ){
      std::cout << "Norm of u_ptr is not one." << std::endl;
      exit(EXIT_FAILURE);
    }

    // reassign pointer XBraid will use
    *u_ptr = (braid_Vector)u;
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Free(braid_Vector u)
  {
#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Freeing XBraid vectors" << std::endl;
    }
#endif
    my_vector *u_ = (my_vector*) u;
    delete u_;

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Sum(braid_Real alpha,
                       braid_Vector x,
                       braid_Real beta,
                       braid_Vector y)
  {
    // Keep track of the number of times this has been called, just in case we
    // want to track where in the algorithm we are.
    static int sum_count = 0;
    my_vector *x_ = (my_vector *) x;
    my_vector *y_ = (my_vector *) y;

#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Summing XBraid vectors" << std::endl;
      std::cout << alpha << "x + " << beta << "y" << std::endl;
    }
#endif

    ryujin::sadd(y_->U, beta, alpha, x_->U);

    sum_count++;

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::SpatialNorm(braid_Vector u, braid_Real *norm_ptr)
  {
#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Calculating XBraid vector spatial norm" << std::endl;
    }
#endif

    my_vector *u_ = (my_vector *)u;
    *norm_ptr = std::get<0>(u_->U).l2_norm();

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Access(braid_Vector u, BraidAccessStatus &astatus)
  {
    my_vector *u_ = (my_vector *) u;

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
        mgrit_functions::enforce_physicality_bounds<Description, dim, Number>(
            *u_, finest_level, *this, t);
#ifdef CHECK_BOUNDS
        test_physicality(
            std::get<0>(u_->U), 0, "my_Access: u when caller is FInterp_Projection.");
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
          print_solution(u_->U, t, finest_level /*level that u lives on*/, fname, false,t_idx);
        }
        if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
          std::cout << "Cycles done: " << mgCycle << std::endl;
        }
        // calculate drag (at end of cycle...)
        dealii::Tensor<1, dim> forces =
            mgrit_functions::calculate_drag_and_lift<Number, Description>(this, *u_, t);
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
        // Do nothing otherwise.
        break;
      }
    }

      return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufSize(braid_Int *size_ptr,
                           BraidBufferStatus &bstatus)
  {
#ifdef DEBUG
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] Buf_size Called" << std::endl;
    }
#endif

    // TODO: answer question about what the buffer size whould be, i think it
    // should be problem_dimension*number of spatial nodes. But there is some
    // question in my mind about if the MPI communication is from this Time
    // Brick (which owns a distributed vector on a few processors) or among the
    // spatial processors. I suspect the former.
    UNUSED(bstatus);

    // no vector can be bigger than this, so we are very conservative.
    int size = n_fine_dofs * problem_dimension;
    *size_ptr =
        (size + 1) * sizeof(Number); //+1 is for the size of the buffers being
                                     // stored in the first component.
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "Size in bytes of the Number: " << sizeof(Number)
                << std::endl;
      std::cout << "Problem_dimension: " << problem_dimension
                << " n_dofs: " << n_fine_dofs << std::endl;
      std::cout << "buf_size: " << *size_ptr << std::endl;
    }

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufPack(braid_Vector u,
                           void *buffer,
                           BraidBufferStatus &bstatus)
  {
    my_vector *u_ = (my_vector *) u;
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] BufPack Called" << std::endl;
    }

    Number *dbuffer = (Number *)buffer;
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
            std::get<0>(u_->U).local_element(problem_dimension * node + component);
        Assert(!std::isnan(
                   std::get<0>(u_->U).local_element(problem_dimension * node + component)),
               dealii::ExcMessage(
                   "The vector you are trying to pack has a NaN in it at "
                   "component " +
                   std::to_string(problem_dimension * node + component)));
      }
    }
    bstatus.SetSize(
        (buf_size + 1) *
        sizeof(Number)); // set the number of bytes stored in this buffer (TODO:
                         // this is off since the dbuffer[0] is a integer.)
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufUnpack(void *buffer,
                             braid_Vector *u_ptr,
                             BraidBufferStatus &bstatus)
  {
    if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
      std::cout << "[INFO] BufUnpack Called" << std::endl;
    }

    UNUSED(bstatus);

    Number *dbuffer = (Number *)buffer;
    // todo: use this for a range check. Make sure we are not indexing outside of bounds of the buffer.
    unsigned int buf_size = static_cast<unsigned int>(dbuffer[0]); // TODO: is this dangerous?

    // The vector should be size (dim + 2) X n_dofs at finest level.
    my_vector *u = new (my_vector); // TODO: where does this get deleted? Probably
                                  // wherever owns the u_ptr.
    reinit_to_level(u, finest_level); // each U is at the finest level.

    // unpack the sent data into the right level
    for (unsigned int node = 0; node < n_locally_owned_dofs; node++) {
      // get tensor at node.
      for (unsigned int component = 0; component < problem_dimension;
           ++component) {
        std::get<0>(u->U).local_element(problem_dimension * node + component) =
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
    test_physicality(std::get<0>(u->U), 0, "my_BufUnpack: unpacked vector.");
#endif

    return 0;
  }
}//Namepsace mgrit
