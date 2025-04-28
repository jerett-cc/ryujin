#include <cmath>

#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "euler/description.h"
#include "euler/hyperbolic_system.h"
#include "my_app.h"
#include "state_vector.h"
#include "mgrit_functions.template.h"


const std::string parameters =
  "subsection App\n"
  "  set print_solution_bool = true\n"
  "  set Time Bricks = 16 \n"
  "  set min_num_coarsest_points = 2\n"
  "  set print factor = 2 # how many cpoints do we skip during output?\n"
  "  set Start Time = 0.0\n"
  "  set Stop Time = 2.0\n"
  "  set cfactor = 2 ## 2 is default\n"
  "  set max_iter = 2 # do this many iterations\n"
  "  set use_fmg = false ## use f multigrid cycle\n"
  "  set n relax = 1\n"
  "  set access_level = 3 ## if this dips, then we will not have the projection operation,\n"
  "                       ## which is key for stability \n"
  "		       ## 3 = projection operations happen\n"
  "		       ## 4 = many tau visualizations at all levels\n"
  "  set base name = problem_brick_E_r346_two_cycle_VMG\n"
  "end\n";


/**
 * This tests that interpolation coarse->fine between many levels is the same as
 * interpolating between intermediate levels sequentially.
 */
using StateVector = mgrit::MyApp<double, ryujin::Euler::Description, 2>::StateVector;
using Vector = mgrit::MyVector<double, ryujin::Euler::Description, 2>;
using App = mgrit::MyApp<double, ryujin::Euler::Description, 2>;

constexpr int dim = 2;

void random_initial_values(Vector &U, const int level, const App &app)
{
  auto &u = std::get<0>(U.U);
  const int n_dofs = app.n_locally_owned_at_level(level);
  const int n_components = app.problem_dimension;
  int x = 0;
  for(int i = 0; i < n_dofs; i++)
  {
    auto state = u.get_tensor(i);
    for(int d=0; d<n_components; d++)
      {
	state[d] = sin(x++);
      }

    u.write_tensor(state,i);
  }
};

int main(int argc, char *argv[]){

  const std::vector<int> refinement_levels = {1, 3, 5};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, ryujin::Euler::Description, 2> app(comm_world, comm_world, refinement_levels);

  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);

  // Set up data.
  mgrit::MyVector<double, ryujin::Euler::Description, 2> fineU, fineCopy,
                                                         middleU,
                                                         coarseU, coarseCopy;
  // Now compare interpolation coarse->fine
  app.reinit_to_level(&fineCopy, 0);
  app.reinit_to_level(&fineU, 0);
  app.reinit_to_level(&middleU, 1);
  app.reinit_to_level(&coarseU, 2);

  // Set up coarse level with random data.
  random_initial_values(coarseU, 2, app);

  // now interpolate the coarse to fine.
  app.interpolate_between_levels(fineU, 0, coarseU, 2);
  // and then coarse to the middle, then the middle to the fine copy.
  app.interpolate_between_levels(middleU, 1, coarseU, 2);
  app.interpolate_between_levels(fineCopy, 0, middleU, 1);

  //compare that the single coarse->fine = coarse->middle->fine
  std::get<0>(fineU.U) -= std::get<0>(fineCopy.U);
  auto diff = std::get<0>(fineU.U).l2_norm();

  if (diff < 1e-10)
  {
    std::cout << "Interpolation coarse->fine consistent." << std::endl;
  }
  else
  {
    std::cout << "Coarse->Fine Not OK." << std::endl;
  }

  return 0;
}
