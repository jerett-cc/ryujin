#include <cmath>

#include "discretization.h"
#include "level_structures.h"//for all the objects that are needed for a run.
#include "time_loop.h"
#include <deal.II/base/mpi.h>
#include "navier_stokes/description.h"
#include "euler/hyperbolic_system.h"
#include "level_structures.h" //for all the objects that are needed for a run.
#include "mgrit_functions.template.h"
#include "my_app.h"
#include "state_vector.h"
#include "time_loop.h"
#include <deal.II/base/mpi.h>


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
    "  set access_level = 3\n"
    "  set base name = ./test_interpolate_03_\n"
    "end\n"
    "subsection A - TimeLoop\n"
    "  set enable output full = true\n"
    "end\n"
    "subsection C - Discretization\n"
    " set geometry            = cylinder_mgrit\n"
    " subsection cylinder_mgrit\n"
    "   set height                 = 2\n"
    "   set length                 = 4\n"
    "   set object diameter        = 0.5\n"
    "   set object position        = 0.6\n"
    "   set second object diameter = 0.5\n"
    "  end\n"
    "end\n"
    "subsection J - VTUOutput\n"
    "  set manifolds                  = \n"
    "  set schlieren beta             = 10\n"
    "  set schlieren quantities       = rho\n"
    "  set schlieren recompute bounds = true\n"
    "  set use mpi io                 = true\n"
    "  set vorticity quantities       = \n"
    "  set vtu output quantities      = rho, m_1, m_2, E\n"
    "end\n";


/**
 * Test that a solution remains in the invariant domain (density, entropy, and
 * energy all positive) after interpolation to a coarser level.
 */
using StateVector = mgrit::MyApp<double, ryujin::NavierStokes::Description, 2>::StateVector;
using Vector = mgrit::MyVector<double, ryujin::NavierStokes::Description, 2>;
using App = mgrit::MyApp<double, ryujin::NavierStokes::Description, 2>;

constexpr int dim = 2;

int main(int argc, char *argv[])
{

  const std::vector<int> refinement_levels = {2, 4};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv); // create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, ryujin::NavierStokes::Description, 2> app(comm_world, comm_world, refinement_levels);

  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);

  // Set up data.
  mgrit::MyVector<double, ryujin::NavierStokes::Description, 2> fineU, coarseU;

  // now, we need to initialize each vector to the appropriate level.
  app.reinit_to_level(&fineU, 0);
  app.reinit_to_level(&coarseU, 1);

  std::get<0>(fineU.U) =
      app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  // Make sure that this vector is in the invariant domain everywhere.
  // mgrit_functions:: FIXME: add in the check.

  app.time_loops[0]->change_checkpoint_and_frequency_and_basename(
      false, 0.05, app.base_name);
  app.time_loops[0]->run_with_initial_data(fineU.U, 0.15, 0.0, true);

  // now interpolate the fine to the coarse.
  app.interpolate_between_levels(coarseU, 1, fineU, 0);

  // Check that the new coarse vector is ok, and run it.
  // mgrit_functions:: FIXME: add in the check.
  app.time_loops[1]->change_checkpoint_and_frequency_and_basename(
      false, 0.05, app.base_name + "coarse");
  app.time_loops[1]->run_with_initial_data(coarseU.U, 0.25, 0.15, true);

  // check that coarse remains in invariant domain.
  // mgrit_functions:: FIXME: add in the check.

  return 0;
}
