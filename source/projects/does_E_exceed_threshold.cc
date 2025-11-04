#include "discretization.h"
#include "level_structures.h" //for all the objects that are needed for a run.
#include "mgrit_functions.template.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "navier_stokes/description.h"
#include "euler/hyperbolic_system.h"
#include "navier_stokes/parabolic_system.h"
#include "my_app.h"
#include "state_vector.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>


/**
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
using StateVector = mgrit::MyApp<double, ryujin::NavierStokes::Description, 2>::StateVector;
using App = mgrit::MyApp<double, ryujin::NavierStokes::Description, 2>;

constexpr int dim = 2;

void set_E_at_single_point(const double new_point_E, StateVector &U)
{
  // This test only runs on a single process, so all the data structures should
  // be consistent every time. We will set E on dof=0 to be new_point_E.
  // and then be done.
  auto state = std::get<0>(U).get_tensor((unsigned int)0);
  state[dim + 1] = new_point_E;
  std::get<0>(U).write_tensor(state, 0);
}

int main(int argc, char *argv[])
{

  const std::string prm_name = argv[1];
  const std::vector<int> refinement_levels = {0};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, 1); // create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, ryujin::NavierStokes::Description, 2> app(comm_world, comm_world, refinement_levels);

  app.initialize(prm_name);

  // Set up data.
  mgrit::MyVector<double, ryujin::NavierStokes::Description, 2> my_U;

  // Initialize data needs to be at t = 0.
  ryujin::Vectors::reinit_state_vector<ryujin::NavierStokes::Description>(my_U.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(my_U.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  const double E_threshold = 900.;
  // At this point, we should have no problem E.
  if (mgrit_functions::does_E_exceed_threshold<ryujin::NavierStokes::Description,dim,double>(my_U,
										      app,
										      0,
										      0,
										      0,
										      0,
										      E_threshold,
										      false))
  {
    std::cout << "Problem with initialization. E should not be large." << std::endl;
  } else {
    std::cout << "E OK after initialization." << std::endl;
  }

  // set one of the points to have too large of an internal Energy and check
  // that we see this reflected in the output of does_E_exceed_threshold. Turn
  // off printing so no side effects other than the terminal printing happen.
  set_E_at_single_point(1000., my_U.U);
  // At this point, we should have a problem E.
  if (mgrit_functions::does_E_exceed_threshold<ryujin::NavierStokes::Description,dim,double>(my_U,
										      app,
										      0,
										      0,
										      0,
										      0,
										      E_threshold,
										      true))
  {
    std::cout << "Did Exceed, see above for where." << std::endl;
  } else {
    std::cout << "Problem, E should have exceeded at a point." << std::endl;
  }

  return 0;
}
