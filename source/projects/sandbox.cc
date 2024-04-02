#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"




/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
int main(int argc, char *argv[]){

  const std::string prm_name = argv[1];
  const std::string restart_fname = argv[2];
  const double tstart = std::stof(argv[3]);
  const double tstop  = std::stof(argv[4]);
  const int refinement = std::stoi(argv[5]);

  std::cout << "Restarting computation with file " << restart_fname << "\nending at time t in [" << tstart << ", " <<  tstop << "]." << std::endl;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  ryujin::mgrit::LevelStructures<typename ryujin::Euler::Description, 2, double> ls(comm_world, refinement);
  ryujin::TimeLoop<typename ryujin::Euler::Description, 2, double> tl(comm_world, ls);
  std::cout << "Initializing with prm = " + prm_name << std::endl;
  dealii::ParameterAcceptor::initialize(prm_name);

  ls.prepare();

  tl.change_base_name(restart_fname);
  //now that we have the data, we call the run function
  tl.run(tstart);

  return 1;
}