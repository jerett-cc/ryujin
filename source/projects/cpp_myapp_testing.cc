#include <my_app.h>
#include <deal.II/base/mpi.h>

#include "euler/description.h"

#include <string>

int main(int argc, char* argv[])
{
  using Description = ryujin::Euler::Description;
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
  Assert(argc > 1/*program*/ + 1/*parameter file*/ + 1/*px*/ + 1/*at least one level refinement*/,
         dealii::ExcMessage("You must provide the program with a parameter file, a number of spatial processors, "
              "and a multigrid hierarcy. Here, the number of additional parameters needed is at least:" + std::to_string(4-argc)));
  const std::string prm_name(argv[1]);// prm file
  const int px = std::stoi(argv[2]);  // number of processors to use in space
  std::vector<int> refinement_levels(argc - 3);// the vector of refinement levels are equal to the number of remaining arguments, set to argc-3, where 3 is the number of arguments needed before the mg_hierarchy
  for(int i = 3; i < argc; i++)
    refinement_levels[i-3] = std::stoi(argv[i]);


  for(const auto entry: refinement_levels)
    std::cout << entry << std::endl;
  //split the object into the number of time processors, and the number of spatial processors per time chunk.
  MPI_Comm comm_x, comm_t;
  std::cout << "px: " << px << std::endl;

  /**
   * Split WORLD into a time brick for each processor, with a specified number of processors for each to do the spatial MPI.
   * The number of time bricks is equal to NumberProcessorsOnSystem/px    //FIXME: is this true??
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

  std::cout << "ntime in app: " << app.ntime << std::endl;
  BraidCore core(MPI_COMM_WORLD, &app);
  core.SetMaxLevels(app.refinement_levels.size());
  core.SetPrintLevel(3);
  core.SetAbsTol(1.0e-2);
  core.SetCFactor(-1, app.cfactor);
  core.SetPrintFile("braid_debug.txt");
  core.SetAccessLevel(app.access_level);
  core.SetNRelax(-1, app.n_relax);
  core.SetMaxIter(app.max_iter);
  core.SetSeqSoln(0);

  std::cout << "Before braid drive." << std::endl;

  // Run Simulation
  core.Drive();
}