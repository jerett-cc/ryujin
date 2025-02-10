#include <my_app.h>
#include <deal.II/base/mpi.h>

#include "euler/description.h"

#include <string>

int main(int argc, char* argv[])
{
  // TODO: make this a parameter file option or a cmd line option.
  using Description = ryujin::Euler::Description;
  //scoped MPI object, no need to call finalize at the end.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);  //create objects
  MPI_Comm comm_world = MPI_COMM_WORLD;//create MPI_object
  //set up app and all underlying data, initialize parameters
  //parse command line parameters, order should be file name, parameter file, px, then the mg hierarcy, i.e. list of refinement levels.
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
  app_0.write_coarse_points("./initial_coarse/"+app_0.base_name);

  // std::vector<NUMBER>c_points =  app.c_points();
  // // Print out the vector
  // for (auto n : c_points)
  //       std::cout << n << ' ';
  //   std::cout << '\n';

}
