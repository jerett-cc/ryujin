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


MyApp::MyApp(const MPI_Comm comm_x,
             const MPI_Comm comm_t,
             const std::vector<unsigned int> a_refinement_levels)
    : BraidApp(comm_t)
    , ParameterAcceptor("/App")
    , comm_x(comm_x)
    , levels(a_refinement_levels.size())
    , refinement_levels(a_refinement_levels)
    , time_loops(a_refinement_levels.size())
    , finest_index(0) // for XBRAID, the finest level is always 0.
{
  coarsest_index = refinement_levels.size() - 1;
  print_solution = false;
  add_parameter("print_solution",
                print_solution,
                "Optional to print the solution in Init and Access.");
  ntime = 100; // This is the default value in BraidApp(...)
  add_parameter("Time Bricks",
                num_time,
                "Number of time bricks total on the fine level.");
  tstart = 0.0;
  add_parameter("Start Time", global_tstart);
  tstop = 5.0;
  add_parameter("Stop Time", global_tstop);
  cfactor = 2; // The default coarsening from level to level is 2, which matches
               // that of XBRAID.
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
  add_parameter("access_level",
                access_level,
                "The level of checking that access will do. 1 is at the end of "
                "the whole simulation, "
                "2 is every cycle, 3 is each interpolation and restriction and "
                "every function.");
};


void MyApp::initialize(std::string prm_file)
{
  
}

braid_Int MyApp::Step(braid_Vector u,
                      braid_Vector ustop,
                      braid_Vector fstop,
                      braid_StepStatus &pstatus)
{
  //this variable is used for writing data to
  //different files during the parallel computations.
  //is passed to run_with_initial_data
  static unsigned int num_step_calls = 0;

  //grab the MG level for this step
  int level;
  BraidStepStatus step_status(pstatus);
  step_status.GetLevel(&level);
  
// #ifdef CHECK_BOUNDS
//   // Test that the incoming vector is physical at the fine level.
//   test_physicality<braid_Vector, 2>(u, app, 0, "before interpolation.");
// #endif

  //use a macro to get rid of some unused variables to avoid -Wall messages
  UNUSED(ustop);
  UNUSED(fstop);
  //grab the start time and end time
  braid_Real brick_tstart;
  braid_Real brick_tstop;
  step_status.GetTstartTstop(&brick_tstart, &brick_tstop);

  if (1/*this was a mpi comm id check before FIXME*/) 
  {
    std::cout << "[INFO] Stepping on level: " + std::to_string(level)+ "\non interval: [" +std::to_string(tstart)+ ", "+std::to_string(tstop)+ "]\n"
      + "total step call number " +std::to_string(num_step_calls) << std::endl;
  }

  std::string fname = "step" + std::to_string(num_step_calls)+ "_cycle" + std::to_string(n_cycles)+ "_level_" + std::to_string(level)
      +"_interval_[" +std::to_string(brick_tstart)+ "_"+std::to_string(brick_tstop)+ "]";

  //translate the fine level u coming in to the coarse level
  //this uses a function from DEALII interpolate to different mesh

  //new, coarse vector
  MyVector u_to_step;
//   reinit_to_level(&u_to_step, app, level);

  //interpolate between levels, put data from u (fine level) onto the u_to_step (coarse level)
//   interpolate_between_levels(u_to_step, level, *u, 0, app);

// #ifdef CHECK_BOUNDS
//   // Test physicality of interpolated vector.
//   test_physicality<my_Vector*, 2>(&u_to_step, app, level, "before step.");
// #endif

  if(print_solution)
    // print_solution(u_to_step.U, app, tstart/*this is a big problem, not knowing what time we are summing at*/, level/*level, always needs to be zero, to be fixed*/, fname, false, app->n_cycles);

  if(level == 1 && std::abs(tstart - 3.125) < 1e-6 && std::abs(tstop - 3.4375) < 1e-6 && num_step_calls==132)
    ryujin::Checkpointing::write_checkpoint(
            *(app->levels[level]->offline_data), fname, u_to_step.U, tstart, num_step_calls, app->comm_x);
  //step the function on this level
  time_loops[level]->change_base_name(fname);
  time_loops[level]->run_with_initial_data(u_to_step.U, brick_tstop, brick_tstart, false/*print every step of this simulation*/);

// #ifdef CHECK_BOUNDS
//   // Test physicality of vector after it has been stepped.
//   test_physicality<my_Vector*, 2>(&u_to_step, app, level, "after step.");
// #endif

//   double norm = u_to_step.U.l1_norm();

//   if(!dealii::numbers::is_finite(norm)){
//     // ryujin::Checkpointing::write_checkpoint(
//     //         *(app->levels[level]->offline_data), fname, u_to_step.U, tstop, num_step_calls, app->comm_x);
//     std::cout << "nan in file " << fname << std::endl;
//     std::cout << "Norm was " << norm << std::endl; 
//     // exit(EXIT_FAILURE);
//   }
//   //interpolate this back to the fine level
//   interpolate_between_levels(*u,0,u_to_step,level, app);

// #ifdef CHECK_BOUNDS
//   // Test physicality of interpolated vector on fine level, after the step.
//   test_physicality<braid_Vector,2>(u, app, 0, "after step, after interpolation.");
// #endif

  std::string fname_post = "FcOnLevel_" + std::to_string(level)+ "on_interval_[" +std::to_string(brick_tstart)+ "_"+std::to_string(brick_tstop)+ "]";
  if(print_solution)
    // print_solution(u->U, app, tstop, 0/*level, always needs to be zero, to be fixed*/, fname_post, false, app->n_cycles);

  num_step_calls++;
  //done.
  return 0;
};
