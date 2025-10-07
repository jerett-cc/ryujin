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

/**
 * Change rounding mode on X86-64 architecture: Denormals are flushed to
 * zero to avoid computing on denormals which can slow down computations
 * significantly.
 */
void flush_denormals_to_zero()
{
#if defined(DENORMALS_ARE_ZERO) && defined(__x86_64)
#define MXCSR_DAZ (1 << 6)  /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15) /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr(mxcsr);
#endif
}

/**
 * Set up thread pools and obey thread limits:
 */
void set_thread_limit(const MPI_Comm &mpi_communicator [[maybe_unused]])
{
  unsigned int n_threads = 1;
#ifdef WITH_OPENMP
  const unsigned int n_threads_omp = omp_get_thread_limit();
  const unsigned int n_threads_dealii = dealii::MultithreadInfo::n_threads();
  n_threads = std::min(n_threads_omp, n_threads_dealii);
  omp_set_num_threads(n_threads);
#endif

  dealii::MultithreadInfo::set_thread_limit(n_threads);
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Using " +
                     std::to_string(dealii::MultithreadInfo::n_threads()) +
                     " threads."
              << std::endl;
}

int main(int argc, char *argv[])
{
  // feenableexcept(FE_DIVBYZERO | FE_INVALID);
#ifdef ENABLE_FPE
  feenableexcept(FE_DIVBYZERO | FE_INVALID);
#endif
  flush_denormals_to_zero();

  // TODO: make this a parameter file option or a cmd line option.
  using Description = ryujin::Euler::Description;
  LSAN_DISABLE;
  // scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv);                      // create objects
  MPI_Comm comm_world = MPI_COMM_WORLD; // create MPI_object
  set_thread_limit(comm_world);
  LSAN_ENABLE;
  LIKWID_INIT;

  dealii::ConditionalOStream pout(std::cout);
  pout.set_condition(dealii::Utilities::MPI::this_mpi_process(comm_world) == 0);

  // set up app and all underlying data, initialize parameters
  // parse command line parameters, order should be file name, parameter file,
  // px, then the mg hierarcy, i.e. list of refinement levels.
  Assert(argc >= 1 /*program*/ + 1 /*parameter file*/ + 1 /*px*/ +
                     1 /*at least one level refinement*/,
         dealii::ExcMessage("You must provide the program with a parameter "
                            "file, a number of spatial processors, "
                            "and a multigrid hierarcy. Here, the number of "
                            "additional parameters needed is at least:" +
                            std::to_string(4 - argc)));
  const std::string prm_name(argv[1]); // prm file
  const int px = std::stoi(argv[2]);   // number of processors to use in space
  std::vector<int> refinement_levels(
      argc - 3); // the vector of refinement levels are equal to the number of
                 // remaining arguments, set to argc-3, where 3 is the number of
                 // arguments needed before the mg_hierarchy
  for (int i = 3; i < argc; i++)
    refinement_levels[i - 3] = std::stoi(argv[i]);


  for (const auto entry : refinement_levels)
    pout << entry << std::endl;
  // split the object into the number of time processors, and the number of
  // spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  pout << "px: " << px << std::endl;

  /**
   * Split WORLD into a time brick for each processor, with a specified number
   * of processors for each to do the spatial MPI. The number of time bricks is
   * equal to NumberProcessorsOnSystem/px    //FIXME: is this true??
   */
  Assert(dealii::Utilities::MPI::n_mpi_processes(comm_world) % px == 0,
         dealii::ExcMessage(
             "You are trying to divide world into a number of spatial "
             "processors per time brick that will cause MPI to stall. The "
             "variable px needs to divide the number of processors total."));
  braid_SplitCommworld(&comm_world,
                       px /*the number of spatial processors per time brick*/,
                       &comm_x,
                       &comm_t);

  mgrit::MyApp<NUMBER, Description, 2> app(comm_x, comm_t, refinement_levels);
  app.initialize(prm_name);

  // std::vector<NUMBER>c_points =  app.c_points();
  // // Print out the vector
  // for (auto n : c_points)
  //       pout << n << ' ';
  //   pout << '\n';

  pout << "ntime in app: " << app.ntime << std::endl;
  BraidCore core(MPI_COMM_WORLD, &app);
  core.SetMaxLevels(app.max_levels);
  core.SetPrintLevel(3);
  core.SetAbsTol(1.0e-2);
  core.SetCFactor(-1, app.cfactor);
  core.SetPrintFile("braid_debug.txt");
  core.SetAccessLevel(app.access_level);
  core.SetNRelax(-1, app.n_relax);
  core.SetMaxIter(app.max_iter);
  core.SetSeqSoln(app.use_sequential_solution);
  core.SetSkip(app.skip_first_down_cycle);

  pout << "Before braid drive." << std::endl;

  // Run Simulation
  core.Drive();
  app.print_times();
  // app.print_bricks_relaxation_count();
  LIKWID_CLOSE;
  LSAN_DISABLE;

  return 0;
}
