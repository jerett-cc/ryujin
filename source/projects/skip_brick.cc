#include "discretization.h"
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "level_structures.h" //for all the objects that are needed for a run.
#include "my_app.h"
#include "state_vector.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>


/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
 */
int main(int argc, char *argv[])
{

  const std::string prm_name = "skip_brick.prm";
  const std::vector<int> refinement_levels = {0, 1, 2};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, 1); // create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;

  mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2> app(
      comm_world, comm_world, refinement_levels);

  app.initialize(prm_name);

  using StateVector =
      mgrit::MyApp<NUMBER, ryujin::Euler::Description, 2>::StateVector;

  // Set up data.
  StateVector U;

  // Initialize data needs to be at t = 0.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(
      U, *(app.levels[0]->offline_data));
  std::get<0>(U) =
      app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  // pretend that we are on cycle 2, check that all bricks which should not step
  // on this cycle don't;

  const braid_Int cycle = 2;
  const braid_Int n_bricks = app.ntime / app.cfactor;
  app.n_relax = 1;

  std::cout << "Number of Bricks expected = " << 12
            << " Number from prm = " << n_bricks << std::endl;

  // At start of cycle 2, with n_relaxations = 1 and cfactor=2 with 12 bricks,
  // the following levels exist where X means that the previous C-point is
  // exact:
  // 0  1  2    3   4   5  6   7  8   9  10  11 12 ->t_idx
  // X |X ||_X_|_X_||__|__||__|__||__|__||__|__|| L=0
  // __X__||___X___||_____||_____||_____||_____|| L=1
  // ______X______ || ___________|| ___________|| L=2

  // FIXME: fix this test and change the test in tests/

  std::cout << ((app.previous_cpoint_is_exact(0, cycle)) ? "OK" : "Not OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(1, cycle)) ? "OK" : "Not OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(2, cycle)) ? "OK" : "Not OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(3, cycle)) ? "OK" : "Not OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(4, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(5, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(6, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(7, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(8, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(9, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(10, cycle)) ? "Not OK" : "OK")
            << std::endl;
  std::cout << ((app.previous_cpoint_is_exact(11, cycle)) ? "Not OK" : "OK")
            << std::endl;

  return 0;
}
