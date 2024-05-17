#include "my_app.h"
#include <deal.II/base/mpi.h>


int main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  braid_MPI_Comm world = MPI_COMM_WORLD;
  mgrit::MyApp app(world, world, std::vector<int>({0, 1}));
  app.initialize("test.prm");

  braid_Vector *v = new(braid_Vector);
  app.Init(2.5, v);

  std::string fname = "./test_init_at_" + std::to_string(2.5);
  mgrit::MyVector *v_ = (mgrit::MyVector *)v;
  app.print_solution(v_->U, 2.5, app.finest_level, fname, false, -1);
  delete v;
}
