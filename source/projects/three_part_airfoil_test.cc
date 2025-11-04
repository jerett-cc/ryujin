#include <deal.II/base/mpi.h>
#include <my_app.h>

#include <cfenv> // for floating point exceptions

#include "euler/description.h"
#include "introspection.h"
#include "time_integrator.template.h"

#include <string>

// Some thread includes.
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_out.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <filesystem>


const std::string parameters =
    "subsection App\n"
    "  set print_solution_bool = true\n"
    "  set Time Bricks = 16\n"
    "  set min_num_coarsest_points = 2\n"
    "  set print factor = 2\n"
    "  set Start Time = 0.0\n"
    "  set Stop Time = 2.0\n"
    "  set cfactor = 2\n"
    "  set max_iter = 2\n"
    "  set use_fmg = false\n"
    "  set n relax = 1\n"
    "  set access_level = 3\n"
    "  set base name = ./test_mgrit_same_mesh\n"
    "end\n"
    "subsection C - Discretization\n"
    " set geometry            = three part airfoil\n"
    " subsection three part airfoil\n"
    "   set three part wing file location = ./wing.msh\n"
    "  end\n"
    "end\n";

int main(int argc, char *argv[])
{

  // TODO: make this a parameter file option or a cmd line option.
  using Description = ryujin::Euler::Description;
  LSAN_DISABLE;
  // scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv);                      // create objects
  MPI_Comm comm_world = MPI_COMM_WORLD; // create MPI_object
  LSAN_ENABLE;
  LIKWID_INIT;

  dealii::ConditionalOStream pout(std::cout);
  pout.set_condition(dealii::Utilities::MPI::this_mpi_process(comm_world) == 0);

  const int px = 1;         // number of processors to use in space
  const int refinement = 0; // spatial refinement
  // a vector of names for the time integrator used on each level
  std::vector<std::string> integrator_levels({"erk 11"});

  for (const auto &entry : integrator_levels)
    pout << entry << std::endl;
  // split the object into the number of time processors, and the number of
  // spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  pout << "px: " << px << std::endl;

  braid_SplitCommworld(&comm_world,
                       px /*the number of spatial processors per time brick*/,
                       &comm_x,
                       &comm_t);

  mgrit::MyApp<NUMBER, Description, 2> app(
      comm_x, comm_t, integrator_levels, refinement);
  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);

  pout << "Done initializing app." << std::endl;

  // Set up data.
  mgrit::MyVector<double, ryujin::Euler::Description, 2> my_U;

  // Initialize data my_U at t = 0.
  ryujin::Vectors::reinit_state_vector<ryujin::Euler::Description>(
      my_U.U, *(app.levels[0]->offline_data));
  // Set the solution to 0.
  ryujin::set_all_entries(my_U.U, 0);

  // print mesh.
  pout << "Writing solution" << std::endl;
  app.print_solution(my_U.U, 0.0, app.finest_level, "wing", 0);

  // Finally, produce a graphical representation of the mesh to an output
  // file:
  pout << "Writing wing_mesh" << std::endl;
  std::ofstream out("wing_mesh.vtu");
  dealii::GridOut grid_out;
  grid_out.write_vtu(
      app.levels[0]->offline_data->discretization().triangulation(), out);

  LIKWID_CLOSE;
  LSAN_DISABLE;

  return 0;
}
