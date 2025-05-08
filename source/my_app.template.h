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
#include <set>

//ryujin includes
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "geometry_cylinder.h"
#include "discretization.h"
#include "hyperbolic_system.h"
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
#include "mpi_ensemble.h"
#include "mpi_ensemble_container.h"

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
      , mpi_ensemble_x(std::make_shared<
		         ryujin::MPIEnsemble>(comm_x,
					      /*n_ensembles=*/1,
					      /*global_synchronization=*/false))
      , levels(a_refinement_levels.size())
      , refinement_levels(a_refinement_levels)
      , time_loops(a_refinement_levels.size())
      , finest_level(0) // for XBRAID, the finest level is always 0.
      , discretization_vec(1)
      , offline_data_vec(1) // initialize this with only one level, will resize later.
      , pout(std::cout)
  {
    coarsest_level = refinement_levels.size() - 1;
    print_solution_bool = false;
    add_parameter("print_solution_bool",
                  print_solution_bool,
                  "Optional to print the solution in Init and Access.");
    num_bricks = 10; // This is the default value in BraidApp(...)
    add_parameter("Time Bricks",
                  num_bricks,
                  "Number of time bricks total on the fine level.");
    tstart = 0.0;
    add_parameter("Start Time", tstart);
    tstop = 5.0;
    add_parameter("Stop Time", tstop);
    cfactor = 2; // The default coarsening from level to level is 2, which
                 // matches that of XBRAID.
    add_parameter(
        "cfactor", cfactor, "The coarsening factor between time levels.");
    max_iter = num_bricks; // In theory, mgrit should converge after the number of
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
    base_name = "mgrit";
    add_parameter("base name",
        base_name,
        "The name used in printing in ryujin");
    minimal_tpoints_coarsest_level = 3;
    add_parameter("min_num_coarsest_points",
		  minimal_tpoints_coarsest_level,
		  "The smallest number of allowable time points the "
		  "user defines on the most coarse level. Must be >=1.");
    print_factor = 1;
    add_parameter("print factor",
		  print_factor,
		  "Contols how many cpoints to output, 1 means all, 2 means skip every other, "
		  "and so on.");

    storage_name = "./initial_coarse/";
    add_parameter("the location of where you wish to store the initial guesses",
		  storage_name,
		  "if you want it to be in the run directory, simply use './name_you_desire/' ");

    // drag_history()//TODO: how to init? need this to have max_iter num of vectors, each of size n_coarse_points(at least the ones I am printing.)
  };

  template<typename Number, typename Description, int dim>
  MyApp<Number, Description, dim>::~MyApp(){
  };

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::initialize(const std::string &prm_file)
  {
    std::ifstream prm_stream (prm_file);
    initialize(prm_stream);
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::initialize(std::istream &prm_stream)
  {
    ryujin::Scope scope(computing_timer, "initialize");

    // Set the condition for the pout, only output on p0 in the global communicator.
    // TODO: add a parameter 'print_to_terminal' or something like that and replace
    // the condition below with an p0=0 $$ print_to_terminal.
    pout.set_condition(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    
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
      if (std::find(refinement_levels.begin(),
                    refinement_levels.end(),
                    lvl) != refinement_levels.end()) {
        discretization_vec[most_refinement-lvl] = levels[iter]->discretization;
        offline_data_vec[most_refinement-lvl] = levels[iter]->offline_data;
        level_map[iter] =
            most_refinement -
            lvl; // The lvl, if it is one we care about for computations, lives
                 // in the offline_data_vec at index most_refinement-lvl, which
                 // in principle is not the same at lvl.
        iter++;
      } else {
        discretization_vec[most_refinement-lvl] = std::make_shared<DiscretizationType>(
            *mpi_ensemble_x, lvl, "/C - Discretization");
        offline_data_vec[most_refinement-lvl] = std::make_shared<OfflineDataType>(*mpi_ensemble_x, 
                                                                *discretization_vec[most_refinement-lvl],
                                                                "/OfflineData");
      }
    }
    // now that levels are all created, we parse the parameter file.
    dealii::ParameterAcceptor::initialize(prm_stream);

    // all parameters defined, we can now call all objects prepare function.
    prepare_mg_objects();

    // Prepare the additional offline_data and discretizations.
    pout << "[INFO] Preparing additional offline_data and "
      "discretization for interpolation purposes."
	 << std::endl;
    for (int lvl = most_refinement; lvl >= least_refinement; lvl--) {
      // If we don't find this level already set up, we prepare it.
      if (std::find(refinement_levels.begin(),
                    refinement_levels.end(),
                    lvl) == refinement_levels.end())
      {  
        discretization_vec[most_refinement-lvl]->prepare(base_name);
	const unsigned int n_parabolic_state = offline_data_vec[most_refinement-lvl]
	  ->n_parabolic_state_vectors();
        offline_data_vec[most_refinement-lvl]->prepare(problem_dimension,
						       n_precomputed_values,
						       n_parabolic_state);
      }
    }
    pout << "Additional offline_data and discretization prepared" << std::endl;
    // Set the number of time points based on the number of bricks.
    for(braid_Int l = 0; l < coarsest_level; l++)
      total_cfactor *= cfactor;
    pout << "Cumulative coarsening by a factor of " << total_cfactor << std::endl;

    Assert(minimal_tpoints_coarsest_level >=1,
	   dealii::ExcMessage("Your choice of " + std::to_string(minimal_tpoints_coarsest_level)+
			      " time points on the coarsest level needs to be >=1,"
			      " which is the default."));
    // Now that we know the total coarsening, we need to determine the ntime variable
    // giving the correct number of coarse time points.
    ntime = num_bricks * total_cfactor * minimal_tpoints_coarsest_level;
    ntime = 40;//TODO: remove me.
    Assert((print_factor >=1 && print_factor < ntime),
	   dealii::ExcMessage("Print factor must be at least one, and less than the number of "
			      "time points total."));

    storage_name = storage_name + base_name;

    n_parabolic_state_vectors = unrefined_level->parabolic_system->get().n_parabolic_state_vectors();
    //   initialized = true; // now the user can access data in app. TODO:
    //   implement a check for getter functions.
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::create_mg_levels()
  {
    // Make the unrefined levels, so that we can use it in the Init()
    // function.
    unrefined_level = std::make_shared<
	                ryujin::mgrit::LevelStructures<Description,
						       dim,
						       Number>>(mpi_ensemble_x,
								0/*no refinement*/);
    pout << "[INFO] Creating Required Level Structures" << std::endl;
    for (unsigned int i = 0; i < refinement_levels.size(); i++) {
      levels[i] = std::make_shared<
          ryujin::mgrit::LevelStructures<Description, dim, Number>>(
          mpi_ensemble_x, refinement_levels[i]);
      time_loops[i] =
          std::make_shared<ryujin::TimeLoop<Description, dim, Number>>(*(levels[i]));
    }
    pout << "All MG levels created" << std::endl;
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::prepare_mg_objects()
  {
    unrefined_level->prepare(base_name);

    pout << "[INFO] Preparing Required Level Structures" << std::endl;
    for (unsigned int lvl = 0; lvl < refinement_levels.size(); lvl++) {
      levels[lvl]->prepare(base_name);

      MPI_Barrier(MPI_COMM_WORLD); // TODO: need this?
    }
    pout << "All MG levels prepared " << std::endl;
    // set the last variables in app.
    n_fine_dofs = levels[0]->offline_data->dof_handler().n_dofs();
    n_locally_owned_dofs = levels[0]->offline_data->n_locally_owned();
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::print_times()
  {
    // Sum across spatial processors, sum across temporal processors, and Print
    // only if we are a specific processor.
    for(auto &it : computing_timer)
    {
      const auto statistics_space_time =
          dealii::Utilities::MPI::min_max_avg(it.second.cpu_time(), mpi_ensemble_x->ensemble_communicator());
      
      pout << "Total time for " << it.first << ": "
	   << std::setprecision(4) << std::fixed << std::setw(9)
	   << statistics_space_time.sum << std::endl;
     
    }
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::print_bricks_relaxation_count()
  {
    // Sum across spatial processors, sum across temporal processors, and Print
    // only if we are a specific processor.
    for(auto &it : f_brick_relaxation_count)
    { 
      pout << "Count for brick " << it.first.first
		  << " on iteration " << it.first.second << ": "
                  << it.second << std::endl;
    }
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::reinit_to_level(my_vector *u, const int level) const
  {
    Assert(levels.size() > static_cast<unsigned int>(level),
           dealii::ExcMessage("The level being reinitialized does not exist."));
    ryujin::Vectors::reinit_state_vector<Description>(u->U, *(levels[level]->offline_data));
  }
  
  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::interpolate_between_levels(my_vector &to_V,
                                         const int to_level,
                                         const my_vector &from_V,
                                         const int from_level)
  {
    auto& to_v = std::get<0>(to_V.U);
    auto& from_v = std::get<0>(from_V.U);
    Assert(
        (to_v.size() == levels[to_level]->offline_data->dof_handler().n_dofs() *
                            problem_dimension),
        dealii::ExcMessage("Trying to interpolate to a vector and level where "
                           "the n_dofs do not match will not work."));
    Assert(((to_level >= 0) && (from_level >= 0)),
           dealii::ExcMessage("You cannot interpolate to or from a level that "
                              "is negative, all levels are non-negative."));

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
    // TODO: refactor into shared_ptr, then use .get() or something to set then when you use new below.
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
        curr_component.reinit(curr_od->scalar_partitioner(), mpi_ensemble_x->ensemble_communicator());
        next_component.reinit(next_od->scalar_partitioner(), mpi_ensemble_x->ensemble_communicator());

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
        curr_component.reinit(curr_od->scalar_partitioner(), mpi_ensemble_x->ensemble_communicator());
        next_component.reinit(next_od->scalar_partitioner(), mpi_ensemble_x->ensemble_communicator());

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
    pout << "Testing Physicality in location " + where << std::endl;
    const auto hs_view_level =
      levels[level]->hyperbolic_system->get().template view<dim, Number>();

    // Check if the initial condition is admissible as a fluid state. It must
    // have positive density, entropy, and energy.
    for (unsigned int i = 0; i < levels[level]->offline_data->n_locally_owned();
         i++) {
      const bool is_admissible =
          hs_view_level.is_admissible(u.template get_tensor(i));
#ifdef DEBUG
      if (!is_admissible) {
        pout << "The state at index i=" + std::to_string(i) +
                         "is not admissible.\n"
                  << "State: " << u.template get_tensor(i) << std::endl;
      }
#endif
      const bool pressure_no_nans =
          (hs_view_level.pressure(u.template get_tensor(i)) ==
           hs_view_level.pressure(u.template get_tensor(i)));
#ifdef DEBUG
      if (!pressure_no_nans) {
        pout << "Pressure is: "
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
						       const unsigned int t_idx)
  {
    Assert(vector_size_match_level(v, level),
	   dealii::ExcMessage("Vector you want to print on level"
			      + std::to_string(level)
			      + " is does not have the right number of"
			      + " dofs for the level."));
    pout << "printing solution" << std::endl;
    //const auto time_loop = time_loops[level];
    // time_loop->output_wrapper(v, fname, t /*current time*/, t_idx /*brick*/);
    levels[level]->vtu_output->schedule_output(v,
					       fname,
					       t,
					       t_idx,
					       true/*output_full*/,
					       false/*output_cutplanes*/);
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::write_checkpoint(StateVector &v,
							 const double t,
							 const std::string fname,
							 const unsigned int t_idx)
  {
    pout << "printing solution" << std::endl;
    const auto time_loop = time_loops[finest_level];
    Assert(levels[finest_level]->offline_data->hyperbolic_vector_partitioner()
	   == std::get<0>(v).get_partitioner(),
	   dealii::ExcMessage("You cannot write a checkpoint unless the vector you "
			      "wish to write is on the finest level (has the same "
			      "partitioner as the finest level)."));
    time_loop->write_checkpoint_wrapper(v, "./checkpoint_" + fname, t, t_idx);
  }


  template<typename Number, typename Description, int dim>
  unsigned int MyApp<Number, Description, dim>::n_locally_owned_at_level(const int level) const
  {
    return levels[level]->offline_data->n_locally_owned();
  }

  template<typename Number, typename Description, int dim>
  bool MyApp<Number, Description, dim>::brick_converged([[maybe_unused]] const braid_Int level,
							const braid_Int brick,
							const braid_Int iter)
  {
    // We base this test on what sort of relaxation we use. We posit that
    // if FC-relaxation is used, then one brick at each level should be exact, as in Parareal.
    // On the flipside, if FCF-relaxation is used, then two bricks will be converged each iteration
    // on each level. TODO: verify that this is true.

    // TODO: does this depend also on the cycle structure?


    // In all other cases, we default to false, since we need to think more carefully about
    // what bricks are converged.

    switch(n_relax)
    {
      case 1:
	{
	  // FC relaxation
	  return (brick < iter) ? true: false;
	}
      case 2:
	{
	  // FCF relaxation
	  return (brick < 2*iter) ? true : false;
	}
      default:
        return false;
	
    }
  }

  template<typename Number, typename Description, int dim>
  std::vector<Number> MyApp<Number, Description, dim>::c_points()
  {
    // The number of c-points is equal to the number of time points divided by the
    // cfactor.

    braid_Int num_cpoints = ntime/cfactor;
    pout << "ntime = " << ntime << " num_cpoints = " << num_cpoints << std::endl;
    Assert(num_cpoints > 0, dealii::ExcInternalError());
// #ifdef DEBUG
//     // Verify these are the same on the finest level
//     BraidCore Core(MPI_COMM_WORLD, this);
//     Core.
//     _braid_Grid      **grids       = _braid_CoreElt(Core.GetCore(), grids);
//     braid_Int          ncpoints    = _braid_GridElt(grids[finest_level], ncpoints);
//     Assert((ncpoints == num_cpoints),
// 	   dealii::ExcMessage("Used num_cpoints " + std::to_string(num_cpoints)+
// 		      " XBraid ncpoints " + std::to_string(ncpoints) + "differ."));
// #endif

    std::vector<Number> c_points(num_cpoints+1);
    Number dt = (tstop-tstart)/num_cpoints;
    
    for(int i = 0; i < num_cpoints+1; i++)
    {
      c_points[i] = tstart + i*dt;
    }
    
    return c_points;
  }

  template<typename Number, typename Description, int dim>
  void MyApp<Number, Description, dim>::write_coarse_points()
  {
    auto c_point = c_points();
    auto dt = c_point[1]-c_point[0];
    // change some printing parameters
    time_loops[0]->change_checkpoint_and_frequency_and_basename(true, dt, storage_name);
    time_loops[0]->set_use_cycle_in_name(true);
    
    // with the time_loop, run on the coarsest level
    time_loops[0]->set_t_final(tstop);
    time_loops[0]->run(tstart);
  }

  template<typename Number, typename Description, int dim>
  bool MyApp<Number, Description, dim>::vector_size_match_level(const StateVector &v,
								const braid_Int level) const
  {
    return std::get<0>(v).size() == problem_dimension * n_locally_owned_at_level(level);
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

    // grab the start time and end time
    double lvl_tstart;
    double lvl_tstop;
    pstatus.GetTstartTstop(&lvl_tstart, &lvl_tstop);

    // grab the MG level for this step
    int level, t_idx, iter, calling;
    pstatus.GetLevel(&level);
    pstatus.GetTIndex(&t_idx);
    pstatus.GetIter(&iter);
    pstatus.GetCallingFunction(&calling); 


    //TODO: do I need this conditional at all? #F-relaxations differ on each level, and
    //      based on relaxation strategy (FC or FCF, etc.)
    if(level == finest_level)
    {
      std::pair<int,int> tidx_iter(t_idx, iter);
      f_brick_relaxation_count[tidx_iter] += 1;
    }
    
    std::string fname = "step" + std::to_string(num_step_calls) + "_cycle" +
      std::to_string(n_cycles) + "_level_" +
      std::to_string(level) + "_interval_[" +
      std::to_string(lvl_tstart) + "_" + std::to_string(lvl_tstop) +"]";
    
    // Start a timer for step::level
    ryujin::Scope scope(computing_timer, "step::" + std::to_string(level));
    
    bool fails = false;
    
#ifdef DEBUG
    fails = fails ||
	mgrit_functions::state_admissible_everywhere(*u_,
						     finest_level,
						     *this,
						     lvl_tstart,
						     calling);
    if(fails)
      print_solution(u_->U,
		       lvl_tstart, finest_level, fname
		     +"not_admissible_before_enforce_physicality_before_step_"
		     +"level_"+std::to_string(level),
		       t_idx);
#endif
    
      // Ensure this is a physical vector.
    mgrit_functions::
        enforce_physicality_bounds<Description, dim, Number>(*u_,
							     finest_level,
							     *this,
							     lvl_tstart,
							     -3);
    
#ifdef DEBUG
      fails = fails ||
	mgrit_functions::state_admissible_everywhere(*u_,
						     finest_level,
						     *this,
						     lvl_tstart,
						     calling);
      if(fails)
	print_solution(u_->U,
		       lvl_tstart, finest_level, fname
		       +"not_admissible_after_enforce_physicality_before_step_"+
		       "level_"+std::to_string(level),
		       t_idx);
	
      pout << "[INFO] Stepping on level: " + std::to_string(level) +
      "\non interval: [" + std::to_string(lvl_tstart) + ", " +
      std::to_string(lvl_tstop) + "]\n" +
      "total step call number " +
      std::to_string(num_step_calls)
	 << std::endl;
#endif

    // use a macro to get rid of some unused variables to avoid -Wall messages
    // TODO: make use of the [[maybe_unused]] tag instead.
    UNUSED(ustop);
    UNUSED(fstop);

    // translate the fine level u coming in to the coarse level
    // this uses a function from DEALII interpolate to different mesh

    // new, coarse vector if levels are not the same, or, a copy of the fine
    // vector, if levels are both finest_level.
    my_vector *u_to_step = new (my_vector);
    reinit_to_level(u_to_step, level);

    // Interpolate between levels, put data from u (fine level) onto the
    // u_to_step (coarse level), if the level is not zero (this is because all
    // the vectors are assumed to be at the finest level spatially.) This allows
    // computations which are naturally faster on the coarser levels, due to a
    // larger mesh size.

    interpolate_between_levels(*u_to_step, level, *u_, 0);
    
    bool print_every_step = false;//TODO Remove these things.
#ifdef DEBUG
    print_every_step = (level == 1) && (t_idx == 3) && (calling == braid_ASCaller_FInterp);
#endif
    // step the function on this level
    // TODO: make sure that the last parameter is set properly, hardcoded
    // is not the best course here.
    time_loops[level]->change_base_name(fname);
    time_loops[level]->run_with_initial_data(
        u_to_step->U,
        lvl_tstop,
        lvl_tstart,
        print_every_step,
	[](const StateVector&, double){},
	print_every_step);//print every step of the integration
    
    // Interpolate the updated state back to the fine level.
    interpolate_between_levels(*u_, 0, *u_to_step, level);

    num_step_calls++;
    delete u_to_step;
   
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int
  MyApp<Number, Description, dim>::Residual(braid_Vector u, braid_Vector r, BraidStepStatus &pstatus)
  {
    /// Does nothing.
    //TODO: replace with [[maybe_unused]]?
    UNUSED(u);
    UNUSED(r);
    UNUSED(pstatus);
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Clone(braid_Vector u, braid_Vector *v_ptr)
  {
#ifdef DEBUG
    pout << "[INFO] Cloning XBraid vectors" << std::endl;
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
    const auto &level_communicator = levels[coarsest_level]->offline_data->dof_handler().get_communicator();
    std::cout << "[INFO] px:" +
      std::to_string(dealii::Utilities::MPI::this_mpi_process(level_communicator))+
      " Initializing XBraid vectors at t="+ std::to_string(t) << std::endl;

    // first, we figure out which C-point this time is. t is an indication. we take the global
    // start and end and calculate the portion of the total time that t is.
    // TODO: is this code safe, in the sense that it will always return basically and interger?
    const braid_Int num_cpoints = ntime/cfactor;
    pout << "Num_cpoints: " << num_cpoints << std::endl;

    // this c_id indicates the number of the checkpoint file we will wish to read
    // so we make a string of where we will find the file.
    const braid_Int c_id = static_cast<braid_Int>(num_cpoints*t/(tstop - tstart));
    const std::string c_file_prefix = storage_name + "-checkpoint" + std::to_string(c_id);

    // We next define a coarse vector at the coarsest level, which will be
    // stepped, then restricted down to the fine level and interpolate the fine
    // initial state into the coarse vector, then interpolates it up to the
    // coarse level and steps.
    std::unique_ptr<my_vector> u = std::make_unique<my_vector>();
    std::unique_ptr<my_vector> temp_coarse = std::make_unique<my_vector>();

    reinit_to_level(u.get(), finest_level);
    reinit_to_level(temp_coarse.get(), coarsest_level);

    // If this is the first brick, we use the initial state of the finest level, not a coarse one.
    // Hence, we skip the load() that happens below, and return early.
    if (c_id == 0)
    {
      Assert(std::abs(t-0.0) < 1e-8,
	     dealii::ExcMessage("Cannot interpolate t=0 conditions onto a vector "
				"that assumes t="+std::to_string(t)));
      // Interpolate t=0 condition.
      std::get<0>(u->U) = levels[finest_level]->initial_values->get().interpolate_hyperbolic_vector(t/*=0.0*/);
      *u_ptr = (braid_Vector)u.release();
      return 0;
    }
    
    //TODO: add a pout to the app so we can use in place of complicated looking
    //      if statements. This will clean up the I/O.
    pout << "Reading file " + c_file_prefix + ".mesh" << std::endl;
    
    // Copy of the triangulation, which we load back in to the triangulation in a hacky
    // way to work around serialization problems.
    auto& unrefined_tria = unrefined_level->discretization->triangulation();
    auto& unrefined_offline_data = *unrefined_level->offline_data;
    
    // load the mesh onto the coarsest mesh. This is needed before the projection
    // can happen below.
    unrefined_tria.load(c_file_prefix+".mesh");

    // Now that a new triangulation is loaded, we need to re-initialize data structures
    // to ensure that we can properly assign data on this triangulation.
    // Because unrefined_tria is always not refined, if the user wants to
    // use my_app where coarsest_level refers to a level with refinement>0,
    // we may cause a bug. This modifies the offline_data, so we need to reset it
    // to the 'unrefined' state before Init() finishes.
    unrefined_offline_data.prepare(problem_dimension,
				   n_precomputed_values,
				   n_parabolic_state_vectors);
    
    /*
     * Read in and broadcast metadata for the coarse data:
     */
    // this is ultimately unused, just needs to be here to read in metadata file
    braid_Int output_cycle = 0;

    unsigned int transfer_handle;
    braid_Real t_in_file = 0.0;
    if (mpi_ensemble_x->world_rank() == 0) {
      std::string meta = c_file_prefix + ".metadata";

      std::ifstream file(meta, std::ios::binary);
      boost::archive::binary_iarchive ia(file);
      ia >> t_in_file >> output_cycle >> transfer_handle;
    }

    int ierr;
    ierr = MPI_Bcast(&transfer_handle,
                     1,
                     MPI_UNSIGNED,
                     0,
                     mpi_ensemble_x->ensemble_communicator());
    AssertThrowMPI(ierr);

    /* Now read in the state vector: */
    unrefined_level->solution_transfer->set_handle(transfer_handle);
    unrefined_level->solution_transfer->project(temp_coarse->U);
    unrefined_level->solution_transfer->reset_handle();

    // Now that we are done, clear the coarse_tria and
    // copy_triangulation from its exact copy. In other words, restore
    // the *invariant* that we have a triangulation and matching
    // DoFHandler that corresponding to the coarse mesh.
    unrefined_tria.clear();
    unrefined_tria.copy_triangulation(unrefined_level->discretization->coarse_triangulation());

    // Since we have clear()'d and copied the triangulation, we need to reset
    // all the data structures in offline_data before we use unrefined_level
    // to do any more work after this function returns.
    unrefined_offline_data.prepare(problem_dimension,
				n_precomputed_values,
				n_parabolic_state_vectors);
    
    // Now interpolate the data we loaded on the coarsest level to the finest level,
    // using the levels data structure, as unrefined_level has done it's work:
    interpolate_between_levels(*u,
			       finest_level,
			       *temp_coarse,
			       coarsest_level);

    // FIXME: the whole cpp interface as awkward use of pointers for the vector objects.
    // See above TODO and relace the poutwith pout here.
    if( !(std::get<0>(u->U).l1_norm()) ){
      pout << "Norm of u_ptr is not one." << std::endl;
      exit(EXIT_FAILURE);
    }

    // reassign pointer XBraid will use by turning ownership of the
    // vector 'u' points to over to 'u_ptr':
    *u_ptr = (braid_Vector)u.release();

    //TODO: replace with a pout.
    pout << "Done with file " << c_file_prefix << std::endl;

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Free(braid_Vector u)
  {
#ifdef DEBUG
    pout << "[INFO] Freeing XBraid vectors" << std::endl;
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
    pout << "[INFO] Summing XBraid vectors" << std::endl;
    pout << alpha << "x + " << beta << "y" << std::endl;
#endif
    
    ryujin::Scope scope(computing_timer, "sum");

    ryujin::sadd(y_->U, beta, alpha, x_->U);

    sum_count++;

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::SpatialNorm(braid_Vector u, braid_Real *norm_ptr)
  {
#ifdef DEBUG
    pout << "[INFO] Calculating XBraid vector spatial norm" << std::endl;
#endif

    my_vector *u_ = (my_vector *)u;
    ryujin::Scope scope(computing_timer, "spatial_norm");

    *norm_ptr = std::get<0>(u_->U).l2_norm();

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::Access(braid_Vector u, BraidAccessStatus &astatus)
  {
    my_vector *u_ = (my_vector *) u;

    braid_Int caller_id;
    braid_Int mgCycle = 0;
    double t = 0;
    braid_Int t_idx;
    braid_Int level;

    // State who is calling, what iteration we are on, and what time t we are accessing.
    astatus.GetCallingFunction(&caller_id);
    astatus.GetIter(&mgCycle);
    astatus.GetT(&t);
    astatus.GetTIndex(&t_idx);
    astatus.GetLevel(&level);

    ryujin::Scope scope(computing_timer, "access::" + std::to_string(level));

    std::string fname = "./" + base_name +"_cycle" + std::to_string(mgCycle);

    
    /*all vectors live on finest level, unless interpolated to a coarser one*/
    bool violates = mgrit_functions::does_E_exceed_threshold(*u_,
							     *this,
							     finest_level,
							     t,
							     t_idx,
							     caller_id,
							     200./*check whenever E is
								   larger than 200*/,
							     true/*print this if does exceed*/,
							     "Elarge_cycle_" + std::to_string(mgCycle)
							     + "forcalling_");
    
    if(violates)
    {
      std::cout << "E for brick " << t_idx << " at t= " << t << " on cycle " << mgCycle
		<< " is bad for caller " << caller_id << std::endl;
      //      mgrit_functions::enforce_physicality_bounds(*u_, finest_level, *this, t);
    }

#ifdef DEBUG
    std::set<int> tau_like_accessors{17,21};
    bool need_assert = !tau_like_accessors.contains(caller_id);
    if(need_assert){
      bool admissible = mgrit_functions::state_admissible_everywhere(*u_,
								     finest_level,
								     *this,
								     t,
								     caller_id);
       
      if(!admissible)  
	{
	  std::cout << "U is not admissible on MG level " << level << " on cycle "
		    << mgCycle << std::endl;  
	  print_solution(u_->U,
			 t,
			 finest_level /*level that every u lives on*/,
			 "./notadmissible_caller"+std::to_string(caller_id),
			 t_idx);
	}
      Assert(admissible,
	     dealii::ExcMessage("A state is not admissible, see file "
				"./notadmissible_caller"+std::to_string(caller_id)));
    }
#endif
    
    switch (caller_id)//FIXME: need switch here? 
    {
      case braid_ASCaller_FAccess: 
      {
	// This function is called at the end of a cycle, if access_level >= 2, and only
	// on the finest level, per XBraid CHANGELOG:Version 2.0.0, 05/25/2016 section.
	Assert(level == finest_level,
	       dealii::ExcMessage("Somehow, braid_ASCaller_FAccess is called on level="
				  + std::to_string(level) + " when we should have been on"+
				  " level=" +std::to_string(finest_level)));
	pout << "[INFO] Access Called" << std::endl;
        pout << "Cycles done: " << mgCycle << std::endl;

	mgrit_functions::enforce_physicality_bounds(*u_, finest_level, *this, t, caller_id);//Needed?
	
	std::cout << "Printing brick " << t_idx << " at t= " << t << " on cycle " << mgCycle
		  << std::endl;
	print_solution(u_->U, t, finest_level /*level that every u lives on*/, fname, t_idx);
	        
        // calculate drag (at end of cycle...)
        dealii::Tensor<1, dim> forces =
            mgrit_functions::calculate_drag_and_lift<Number, Description>(this, *u_, t);

	std::cout << "cycle." + std::to_string(mgCycle) + " drag." +
	                   std::to_string(forces[0]) + " lift." +
	                   std::to_string(forces[1]) + " time." +
	                   std::to_string(t)
	     << std::endl;
	// calculate the conserved quantities in the system, as well as entropy
	mgrit_functions::conserved_and_entropy_in_system<Description,dim,Number>(*u_,
										 this,
										 finest_level,
										 t);
        n_cycles = mgCycle;
        break;
      }
    case braid_ASCaller_FInterp_Projection :
      {
	mgrit_functions::enforce_physicality_bounds(*u_, finest_level, *this, t, caller_id);
      }
      default:
      {
	// Do nothing in a default.
        break;
      }
    }

    return 0;;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufSize(braid_Int *size_ptr,
                           BraidBufferStatus &bstatus)
  {
#ifdef DEBUG
    pout << "[INFO] Buf_size Called" << std::endl;
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
    // if (dealii::Utilities::MPI::this_mpi_process(comm_t) == 0) {
    //   pout << "Size in bytes of the Number: " << sizeof(Number)
    //             << std::endl;
    //   pout << "Problem_dimension: " << problem_dimension
    //             << " n_dofs: " << n_fine_dofs << std::endl;
    //   pout << "buf_size: " << *size_ptr << std::endl;
    // }

    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufPack(braid_Vector u,
                           void *buffer,
                           BraidBufferStatus &bstatus)
  {
    my_vector *u_ = (my_vector *) u;
    pout << "[INFO] BufPack Called" << std::endl;
    
    ryujin::Scope scope(computing_timer, "buf_pack");

    mgrit_functions::
        enforce_physicality_bounds<Description, dim, Number>(
							     *u_, finest_level, *this, 0.0, -2);
    
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
    bstatus.SetSize((buf_size + 1) * sizeof(Number));
    // set the number of bytes stored in this buffer (TODO:
    // this is off since the dbuffer[0] is a integer.)
    pout << "[INFO] BufPack Finished." << std::endl;
    
    return 0;
  }

  template<typename Number, typename Description, int dim>
  braid_Int MyApp<Number, Description, dim>::BufUnpack(void *buffer,
                             braid_Vector *u_ptr,
                             BraidBufferStatus &bstatus)
  {
    pout << "[INFO] BufUnpack Called" << std::endl;

    UNUSED(bstatus);
    ryujin::Scope scope(computing_timer, "buf_unpack");

    Number *dbuffer = (Number *)buffer;
    // todo: use this for a range check. Make sure we are not indexing outside of bounds of the buffer.
    [[maybe_unused]] unsigned int buf_size = static_cast<unsigned int>(dbuffer[0]); // TODO: is this dangerous?

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
