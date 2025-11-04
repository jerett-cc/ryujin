#include "my_app.h"
#include <deal.II/base/mpi.h>

#include "navier_stokes/description.h"


int main(int argc, char *argv[])
{
  using Description = ryujin::NavierStokes::Description;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  braid_MPI_Comm world = MPI_COMM_WORLD;
  mgrit::MyApp<NUMBER, Description, 2> app(
      world, world, std::vector<int>({0, 1}));
  app.initialize("test.prm");

  braid_Vector *v = new (braid_Vector);
  app.Init(2.5, v);
}
