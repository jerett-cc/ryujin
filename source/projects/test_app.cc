#include <iostream>
#include <braid_funcs.h>

int main(int argc, char* argv[]){
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
  Assert(argc > 1/*program*/ + 1/*parameter file*/,
         dealii::ExcMessage("You must provide the program with a parameter file, a number of spatial processors, "
              "and a multigrid hierarcy. Here, the number of additional parameters needed is at least:" + std::to_string(4-argc)));
  const std::string prm_name(argv[1]);// prm file

  // now that we have the communicators, we can create the app, and initialize with the parameter file.
  my_App app(comm_world, comm_world, {argc});
  app.initialize(prm_name);


}