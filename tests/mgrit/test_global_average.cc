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
 * Test that a function which computes a global averaged state which is split equally
 * between two states returns the approximately same as 0.5*(state1 + state2).
 */
constexpr int dim = 2;
using Description = ryujin::Euler::Description;
using StateVector = mgrit::MyApp<double, Description, 2>::StateVector;
using Vector = mgrit::MyVector<double, Description, 2>;
using App = mgrit::MyApp<double, Description, 2>;
using Tensor = dealii::Tensor<1, dim+2, double>;

void two_different_initial_values(Vector &U,
				  const Tensor state1,
				  const Tensor state2,
				  const App &app)
{
  auto &u = std::get<0>(U.U);
  const int n_dof = app.n_locally_owned_at_level(0);// only one level in this test
  for(int i = 0; i < n_dof; i++)
  {
    if(i < n_dof/2)
    {
      u.write_tensor(state1, i);
    } else {
      u.write_tensor(state2, i);
    }
  }
  u.update_ghost_values();
};

int main(int argc, char *argv[]){

  const std::vector<int> refinement_levels = {0};
  
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, Description, 2> app(comm_world,
					   comm_world,
					   refinement_levels);

  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);

  // Set up data.
  mgrit::MyVector<double, Description, 2> U;
  app.reinit_to_level(&U, 0);

  const Tensor state1({1.4, 1, 1, 4});
  const Tensor state2({1, 1, 1, 10});
  const Tensor average({1.2, 1, 1, 7});

  two_different_initial_values(U, state1, state2, app);

  Tensor average_from_function = mgrit_functions::global_average_state(U, 0, app);
  std::cout << "Calculated Average: " << average_from_function;
  std::cout << ", Expected Average: " << average
	    <<  std::endl;
 
  return 0;
}
