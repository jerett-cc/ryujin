#include <deal.II/base/mpi.h>
#include <my_app.h>

#include <cfenv> // for floating point exceptions

#include "euler/description.h"
#include "introspection.h"

#include <string>

// Some thread includes.
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <filesystem>


const std::string parameters = "subsection App\n"
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
                               " set geometry            = cylinder_mgrit\n"
                               " subsection cylinder_mgrit\n"
                               "   set height                 = 2\n"
                               "   set length                 = 4\n"
                               "   set object diameter        = 0.5\n"
                               "   set object position        = 0.6\n"
                               "   set second object diameter = 0.5\n"
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
  const int refinement = 3; // spatial refinement
  // a vector of names for the time integrator used on each level
  std::vector<std::string> integrator_levels({"erk 54", "erk 33", "erk 11"});

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

  pout << "ntime in app: " << app.ntime << std::endl;
  pout << "# LEVELS: " << app.levels.size() << std::endl;
  pout << "  expected: " << 3 << std::endl;
  pout << "mesh info: " << std::endl;
  pout << "expect n_fine_dofs = " << app.n_fine_dofs << std::endl;
  for (auto &level_s : app.levels)
    pout << "  n_dofs: " << level_s->offline_data->dof_handler().n_dofs()
         << std::endl;

  pout << "# discretizations: " << app.discretization_vec.size() << std::endl;
  pout << "# offline_data: " << app.offline_data_vec.size() << std::endl;
  pout << "# refinement levels: " << app.refinement_levels.size() << std::endl;
  for (auto &refinement : app.refinement_levels)
    pout << "  n_global_refinement: " << refinement << std::endl;

  pout << "finest_level: " << app.finest_level << std::endl;
  pout << "coarsest_level: " << app.coarsest_level << std::endl;
  pout << "# time_loops: " << app.time_loops.size() << std::endl;
  auto &convert_to_string =
      dealii::Patterns::Tools::Convert<ryujin::TimeSteppingScheme>::to_string;
  auto &to_pattern =
      dealii::Patterns::Tools::Convert<ryujin::TimeSteppingScheme>::to_pattern;
  pout << "integrators on levels:" << std::endl;
  for (auto &t_loop : app.time_loops)
    pout << "  integrator: "
         << convert_to_string(t_loop->time_integrator().time_stepping_scheme(),
                              *to_pattern())
         << std::endl;

  auto &fine_loop = app.time_loops[0];
  pout << "Making sure that finest level is the right integrator\n"
       << "  expect: erk 54\n"
       << "  actual: "
       << convert_to_string(fine_loop->time_integrator().time_stepping_scheme(),
                            *to_pattern())
       << std::endl;
  pout << "Using same mesh every level "
       << (app.using_same_mesh_every_level ? "True" : "False") << std::endl;

  LIKWID_CLOSE;
  LSAN_DISABLE;

  return 0;
}
