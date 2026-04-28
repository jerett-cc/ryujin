#include <deal.II/base/mpi.h>
#include <my_app.h>

#include "mgrit_description.h"
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
#ifdef WITH_OPENMP
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
#endif
}

int main(int argc, char *argv[])
{
  flush_denormals_to_zero();

  // TODO: make this a parameter file option or a cmd line option.
  using Description = mgrit::Description;

  LSAN_DISABLE;
  // scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv);                      // create objects
  MPI_Comm comm_world = MPI_COMM_WORLD; // create MPI_object
  set_thread_limit(comm_world);
  // set up app and all underlying data, initialize parameters
  // parse command line parameters, order should be file name, parameter file,
  // px, then the mg hierarcy, i.e. list of refinement levels.
  LSAN_ENABLE;

  LIKWID_INIT;
  Assert(argc >= 1 /*program*/ + 1 /*parameter file*/ +
                     1 /*at least one refinement level*/,
         dealii::ExcMessage(
             "You must provide the startup program with a parameter file and a"
             " mesh refinement. "
             "Here, the number of additional parameters needed is at least:" +
             std::to_string(3 - argc)));
  const std::string prm_name(argv[1]); // prm file
  std::vector<int> refinement_levels(
      argc - 2); // the vector of refinement levels are equal to the number of
                 // remaining arguments, set to argc-3, where 3 is the number of
                 // arguments needed before the mg_hierarchy
  for (int i = 2; i < argc; i++)
    refinement_levels[i - 2] = std::stoi(argv[i]);

  std::cout << "prm: " << prm_name
            << " refinement: " << refinement_levels.front() << std::endl;

  // First iteration of MGRIT, initializes the cpoints
  mgrit::MyApp<NUMBER, Description, 1> app_0(
					     comm_world, comm_world, refinement_levels);//ADDBACK! DIM = 2
  app_0.initialize(prm_name);
  app_0.write_coarse_points();

  LIKWID_CLOSE;
  LSAN_DISABLE;
}
