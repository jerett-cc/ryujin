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
 * Right now, this executable runs a simulation equivalent to a ryujin run.
*/
using StateVector = mgrit::MyApp<double, ryujin::Euler::Description, 2>::StateVector;
using App = mgrit::MyApp<double, ryujin::Euler::Description, 2>;

constexpr int dim = 2;

void set_E_and_rho_at_single_point(const double new_point_E, const double new_rho, StateVector &U, bool change_rho = false)
{
  // This test only runs on a single process, so all the data structures should
  // be consistent every time. We will set E on dof=0 to be new_point_E.
  // and then be done.
  auto state = std::get<0>(U).get_tensor((unsigned int)0);
  if(change_rho)
    state[0] = new_rho;
  state[dim+1] = new_point_E;
  std::get<0>(U).write_tensor(state, 0);
}

int main(int argc, char *argv[]){

  const std::vector<int> refinement_levels = {0};
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  mgrit::MyApp<double, ryujin::Euler::Description, 2> app(MPI_COMM_WORLD,
							  MPI_COMM_WORLD,
							  refinement_levels);
  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);

  // Set up data.
  mgrit::MyVector<double, ryujin::Euler::Description, 2> my_U_ok, my_U, my_U_copy, my_U_ref;

  // Initialize data my_U at t = 0.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(my_U.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(my_U.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);
  my_U_copy.U = my_U.U;
  // Initialize my_U_ok and my_U_ref
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(my_U_ok.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(my_U_ok.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(my_U_ref.U,
								   *(app.levels[0]->offline_data));
  std::get<0>(my_U_ref.U) = app.levels[0]->initial_values->get().interpolate_hyperbolic_vector(0.0);

  const double E_threshold = 900.;
  
  // set one of the points to have too large of an internal Energy and a negative density
  // and check that we see this reflected in the output of does_E_exceed_threshold.
  set_E_and_rho_at_single_point(1000.,-2. , my_U.U);
  // At this point, we should have a problem E, and we ought to visualize it.
  if (mgrit_functions::does_E_exceed_threshold<ryujin::Euler::Description,dim,double>(my_U,
										      app,
										      0,
										      0,
										      0,
										      0,
										      E_threshold,
										      false))
  {
    std::cout << "E exceeds threshold like we expect." << std::endl;
  } else {
    std::cout << "Problem, E should have exceeded at some point." << std::endl;
  }

  // Now, we project to stable manifold defined by positive density, and not too large E.
  mgrit_functions::enforce_physicality_bounds(my_U,0,app,0.);

  // Afterwards, we should be not exceeding the threshold anymore, and we nee to visualize the output.
  if (mgrit_functions::
      does_E_exceed_threshold<ryujin::Euler::Description,dim,double>(my_U,
								     app,
								     0,
								     0,
								     0,
								     0,
								     E_threshold,
								     false,
								     "afterenforcephysicality"))
  {
    std::cout << "Projection did not lower E, though it should have." << std::endl;
    // app.print_solution(my_U.U, 0., 0, "./E_exceeds_afterprojection", 0);
  } else {
    std::cout << "Projection has lowered E." << std::endl;
    // app.print_solution(my_U.U, 0., 0, "./E_lowered_afterprojection", 0);
  }

  // we also need to check that the projection is the identity when acting on
  // a vector that is already ok. And that we really have different vector after
  // modifying it and doing a projection.

  const double ref_size = std::get<0>(my_U_ref.U).l1_norm();
  const double modified_size = std::get<0>(my_U.U).l1_norm();
  const double copy_size = std::get<0>(my_U_copy.U).l1_norm();
  // app.print_solution(my_U_ok.U, 0., 0, "./OK_U_before_projection", 0);
  // project an OK vector, expect that we have the same norm as the ref.
  mgrit_functions::enforce_physicality_bounds(my_U_ok,0,app,0.);
  // app.print_solution(my_U_ok.U, 0., 0, "./OK_U_after_projection", 0);
  const double ok_size = std::get<0>(my_U_ok.U).l1_norm();
  std::get<0>(my_U_ok.U).sadd(1., -1., std::get<0>(my_U_ref.U));
  // app.print_solution(my_U_ok.U, 0., 0, "./OK_U__diff_after_projection", 0);
  const double diff_size = std::get<0>(my_U_ok.U).l1_norm();
  // app.print_solution(my_U_ok.U, 0., 0, "./U_ref", 0);
  if ( diff_size >= 1e-5)
  {
    std::cout << "Projection operation NOT identity on physical vector." << std::endl;
  } else {
    std::cout << "Projection operation is identity on physical vector." << std::endl;
  }

  std::cout << "my_Uafter_projection - my_Ucopy=" << std::setprecision(13)
	    << modified_size - copy_size << std::endl;
  std::cout << "my_U_ok_after_projection - my_U_ref=" << std::setprecision(13)
	    << ok_size - ref_size << std::endl;

  
  return 0;
}
