#include <my_app.h>
#include <deal.II/base/mpi.h>

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
  if(dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
}

int main(int argc, char* argv[])
{
  flush_denormals_to_zero();
  
  // TODO: make this a parameter file option or a cmd line option.
  using Description = ryujin::Euler::Description;

  LSAN_DISABLE;
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  set_thread_limit(comm_world);
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
  LSAN_ENABLE;

  LIKWID_INIT;
  Assert(argc >= 1/*program*/ + 1/*parameter file*/ + 1/*minimum refinement*/,
         dealii::ExcMessage("You must provide the startup program with a parameter file and a"
			    " mesh refinement. "
			    "Here, the number of additional parameters needed is at least:"
			    + std::to_string(3-argc)));
  const std::string prm_name(argv[1]);// prm file
  const int refinement = std::stoi(argv[2]);

  std::cout << "prm: " << prm_name << " refinement: " << refinement << std::endl;
  
  // First iteration of MGRIT, initializes the cpoints
  mgrit::MyApp<NUMBER, Description, 2> app_0(comm_world, comm_world, {refinement});
  app_0.initialize(prm_name);
  app_0.write_coarse_points();

  LIKWID_CLOSE;
  LSAN_DISABLE;

}
