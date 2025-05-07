#include <iostream>
#include "my_app.h"
#include "euler/description.h"

const std::string parameters =
"subsection App\n"
  "  set print_solution_bool = true\n"
  "  set Time Bricks = 16\n"
  "  set min_num_coarsest_points = 2\n"
  "  set print factor = 2\n"
  "  set Start Time = 0.0\n"
  "  set Stop Time = 2.0\n"
  "  set cfactor = 3\n"
  "  set max_iter = 2\n"
  "  set use_fmg = false\n"
  "  set n relax = 1\n"
  "  set access_level = 3\n"
  "  set base name = TESTAPP\n"
  "end\n";

int main(int argc, char* argv[]){

  const std::vector<int> refinement_levels = {1, 3, 5};

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);  //create objects
  const MPI_Comm comm_world = MPI_COMM_WORLD;
  
  mgrit::MyApp<double, ryujin::Euler::Description, 2> app(comm_world, comm_world, refinement_levels);
  std::istringstream prm_stream(parameters);
  app.initialize(prm_stream);


  // Make sure that things have been initialized properly. Check that the parameters have been read in.
  std::cout << "cfactor " << ((app.cfactor == 3) ? " OK": " NOT OK") << std::endl;
  std::cout << "finest level " << ((app.finest_level == 0)? " OK": " NOT OK") << std::endl;
  std::cout << "coarsest level " << ((app.coarsest_level == 2)? " OK": " NOT OK") << std::endl;
  std::cout << "num bricks " << ((app.num_bricks == 16)? " OK": " NOT OK") << std::endl;
  std::cout << "access level " << ((app.access_level == 3)? " OK": " NOT OK") << std::endl;
  std::cout << "use_fmg " << ((app.use_fmg == false)? " OK": " NOT OK") << std::endl;
  std::cout << "max iter " << ((app.max_iter == 2)? " OK": " NOT OK") << std::endl;
  std::cout << "base name " << ((app.base_name == "TESTAPP")? " OK": " NOT OK") << std::endl;
  std::cout << "minimal tpoints coarsest level "
	    << ((app.minimal_tpoints_coarsest_level == 2)? " OK": " NOT OK")
	    << std::endl;
  std::cout << "nrelax " << ((app.n_relax == 1)? " OK": " NOT OK") << std::endl;
  std::cout << "tstart " << ((std::abs(app.tstart-0.0) < 1e-8)? " OK": " NOT OK") << std::endl;
  std::cout << "tstop " << ((std::abs(app.tstop-2.0) < 1e-8)? " OK": " NOT OK") << std::endl;
  std::cout << "print factor " << ((app.print_factor == 2)? " OK": " NOT OK") << std::endl;
  std::cout << "print solution bool "
	    << ((app.print_solution_bool == true)? " OK": " NOT OK") << std::endl;

  // Next, test that the data structures at least have the required sizes, not that they have been
  // initialized properly.
  std::cout << "timeloops " << ((app.time_loops.size() == 3)? " OK": " NOT OK") << std::endl;
  std::cout << "level_structures " << ((app.levels.size() == 3)? " OK": " NOT OK") << std::endl;
  std::cout << "discretization levels " << ((app.discretization_vec.size() == 5)? " OK": " NOT OK")
	    << std::endl;
  std::cout << "offline data for interpolation "
	    << ((app.offline_data_vec.size() == 5)? " OK": " NOT OK") << std::endl;
  std::cout << "level map " << ((app.level_map.size() == 3)? " OK": " NOT OK") << std::endl;
}
