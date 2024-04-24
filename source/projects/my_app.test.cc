#include "my_app.h"
#include <deal.II/base/mpi.h>


int main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  braid_MPI_Comm world = MPI_COMM_WORLD;
  mgrit::MyApp app(world, world, std::vector<unsigned int>({0, 1}));
  app.initialize("test.prm");

  MyVector *v = new MyVector;
  app.Init(2.5, &v);

  std::string fname = "./test_init_at_" + std::to_string(2.5);
  app.time_loops[0]->output_wrapper(v->U, fname, 2.5 /*current time*/, 0 /*cycle*/);
}
