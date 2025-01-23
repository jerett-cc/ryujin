#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "my_app.h"
#include "state_vector.h"



/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
int main(int argc, char *argv[]){

  const std::string prm_name = "skip_brick.prm";
  const std::vector<int> refinement_levels = {0, 1, 2};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2> app(comm_world, comm_world, refinement_levels);

  app.initialize(prm_name);

  using StateVector = mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2>::StateVector;

  // Set up data.
  StateVector U;

  // Initialize data needs to be at t = 0.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(U, *(app.levels[0]->offline_data));
  std::get<0>(U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

   //pretend that we are on cycle 2, check that all bricks which should not step on this cycle don't;
  
  const braid_Int cycle    = 2;
  const braid_Int cfactor  = 2;
  const braid_Int n_bricks = app.ntime;
  app.n_relax  = 1;
  app.cfactor = cfactor;
  
  std::cout << "Number of Bricks expected = " << 12 << " Number from prm = " << n_bricks << std::endl;
  
  // On cycle 2, with n_relaxations = 1 and cfactor=2 with 12 bricks, the following levels exist:
  // X |X |__|__|__|__|__|__|__|__|__|__|
  // __X__|__X__|_____|_____|_____|_____|
  // _____X_____|_____X_____|___________|
  // where an X represents that I expect this brick to be converged on this cycle.

  //level 0.
  braid_Int level = 0;
  std::cout << ((app.brick_converged(level, 0, cycle)) ? "OK":"Not OK") << std::endl; 
  std::cout << ((app.brick_converged(level, 1, cycle)) ? "OK":"Not OK") << std::endl;
  std::cout << ((app.brick_converged(level, 2, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 3, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 4, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 5, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 6, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 7, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 8, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 9, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 10, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 11, cycle)) ? "Not OK":"OK") << std::endl;

  //level 1
  level = 1;
  std::cout << ((app.brick_converged(level, 0, cycle)) ? "OK":"Not OK") << std::endl; 
  std::cout << ((app.brick_converged(level, 1, cycle)) ? "OK":"Not OK") << std::endl;
  std::cout << ((app.brick_converged(level, 2, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 3, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 4, cycle)) ? "Not OK":"OK") << std::endl;
  std::cout << ((app.brick_converged(level, 5, cycle)) ? "Not OK":"OK") << std::endl;

  //level 2
  level = 2;
  std::cout << ((app.brick_converged(level, 0, cycle)) ? "OK":"Not OK") << std::endl; 
  std::cout << ((app.brick_converged(level, 1, cycle)) ? "OK":"Not OK") << std::endl;
  std::cout << ((app.brick_converged(level, 2, cycle)) ? "Not OK":"OK") << std::endl;

  return 1;
}
