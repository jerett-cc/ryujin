/*
 * Using ryujin for the higher order Euler simulation of
 * a mach 3 flow around a cylinder.
 */

//includes
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <utility>


//MPI
#include <deal.II/base/mpi.h>

//Braid Implementation
#include <braid_funcs.h>


//todo: change this to a call to something similar to the main ryujin executable. problem_dispach??
int main(int argc, char *argv[])
{
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
  Assert(argc > 1/*program*/ + 1/*parameter file*/ + 1/*px*/ + 1/*at least one level refinement*/,
         dealii::ExcMessage("You must provide the program with a parameter file, a number of spatial processors, "
              "and a multigrid hierarcy. Here, the number of additional parameters needed is at least:" + std::to_string(4-argc)));
  const std::string prm_name(argv[1]);// prm file
  const int px = std::stoi(argv[2]);  // number of processors to use in space
  std::vector<unsigned int> refinement_levels(argc - 3);// the vector of refinement levels are equal to the number of remaining arguments, set to argc-3, where 3 is the number of arguments needed before the mg_hierarchy
  for(int i = 3; i < argc; i++)
    refinement_levels[i-3] = std::stoi(argv[i]);

  for(const auto entry: refinement_levels)
    std::cout << entry << std::endl;
  //split the object into the number of time processors, and the number of spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  std::cout << "px: " << px << std::endl;

  /**
   * Split WORLD into a time brick for each processor, with a specified number of processors for each to do the spatial MPI.
   * The number of time bricks is equal to NumberProcessorsOnSystem/px    //FIXME: is this true??
   */
  Assert(dealii::Utilities::MPI::n_mpi_processes(comm_world) % px == 0,
         dealii::ExcMessage(
             "You are trying to divide world into a number of spatial "
             "processors per time brick that will cause MPI to stall. The "
             "variable px needs to divide the number of processors total."));
  braid_SplitCommworld(&comm_world,
                       px /*the number of spatial processors per time brick*/,
                       &comm_x,
                       &comm_t);

  // now that we have the communicators, we can create the app, and initialize with the parameter file.
  my_App app(comm_x, comm_t, refinement_levels);
  app.initialize(prm_name);

  /* Initialize Braid */
  braid_Core core;
  double tstart = app.global_tstart;
  double tstop = app.global_tstop;
  int ntime = app.num_time;//this should in general be the number of time bricks you want. You need to ensure that px * ntime = TOTAL NUMBER PROCESSORS

  std::cout << "Start: " << tstart << " Stop: " << tstop << " # bricks: " << ntime << std::endl;

  braid_Init(comm_world,
             app.comm_t,
             tstart,
             tstop,
             ntime,
             &app,
             my_Step,
             my_Init,
             my_Clone,
             my_Free,
             my_Sum,
             my_SpatialNorm,
             my_Access,
             my_BufSize,
             my_BufPack,
             my_BufUnpack,
             &core);

  /* Define XBraid parameters
   * See -help message for descriptions */
  unsigned int max_levels = app.refinement_levels.size(); // fixme, cthis is later cast to an int.
  int nrelax = 1; // Default is 1 in XBraid which is FC relaxation(TODO: verify this.), 1 is FCF relaxation, 2 is FCFCF relaxation.
  //      int       skip          = 0;
  double tol = 1e-2;
  int max_iter = app.max_iter;
  // int       min_coarse    = 10;
  // int       scoarsen      = 0;
  // int       res           = 0;
  // int       wrapper_tests = 0;
  int print_level = 3;
  /*access_level=1 only calls my_access at end of simulation*/
  int access_level = 2;
  int use_sequential = 0; //same as XBRAID default, initial guess is from user defined init.

  std::cout << "Parameters for Braid: max_iter= " + std::to_string(max_iter) << std::endl;
  braid_SetPrintLevel(core, print_level);
  braid_SetAccessLevel(core, access_level);
  braid_SetMaxLevels(core, max_levels);
  braid_SetPrintFile(core, "braid_debug.txt");
  braid_SetPrintLevel(core, print_level);
  //             braid_SetMinCoarse( core, min_coarse );
  //             braid_SetSkip(core, skip);
  braid_SetNRelax(core, -1, app.n_relax);
  braid_SetAbsTol(core, tol);
  braid_SetCFactor(core, -1, app.cfactor);
  braid_SetMaxIter(core, app.max_iter);
  braid_SetSeqSoln(core, use_sequential);
  std::cout << "before braid_drive\n";
  braid_Drive(core);

  // Free the memory now that we are done
  braid_Destroy(core);

}
