#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

using namespace dealii;

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  double
  value(const Point<dim> &p, const unsigned int) const
  {
    return p[0] + 2;
  }
};

template <typename MeshType, typename Number>
void
initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec,
                      const MeshType                             &dof_handler,
                      const unsigned int                          level)
{
  IndexSet locally_relevant_dofs;
  if (level == numbers::invalid_unsigned_int)
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
  else
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);


  const parallel::TriangulationBase<MeshType::dimension> *dist_tria =
    dynamic_cast<const parallel::TriangulationBase<MeshType::dimension> *>(
      &(dof_handler.get_triangulation()));

  MPI_Comm comm =
    dist_tria != nullptr ? dist_tria->get_communicator() : MPI_COMM_SELF;

  vec.reinit(level == numbers::invalid_unsigned_int ?
               dof_handler.locally_owned_dofs() :
               dof_handler.locally_owned_mg_dofs(level),
             locally_relevant_dofs,
             comm);
}

template <int dim, typename Number>
void
do_test(const FiniteElement<dim> &fe_fine, const FiniteElement<dim> &fe_coarse)
{
  auto create_fine_grid = [](Triangulation<dim> &tria) {
    GridGenerator::hyper_cube(tria);
    tria.refine_global(2);

    for (auto &cell : tria.active_cell_iterators())
      if (cell->is_active() && cell->center()[0] < 0.5)
        cell->set_refine_flag();
    tria.execute_coarsening_and_refinement();
  };

  auto execute_global_coarsening = [](Triangulation<dim> &tria, const int n_coarsen = 1) {
    for(int n=0; n<n_coarsen; n++)
    {
      for (auto &cell : tria.active_cell_iterators())
        cell->set_coarsen_flag();
      tria.execute_coarsening_and_refinement();
    }
  };

  // create coarse grid
  parallel::distributed::Triangulation<dim> tria_coarse(MPI_COMM_WORLD);
  create_fine_grid(tria_coarse);
  execute_global_coarsening(tria_coarse, 2);

  // create fine grid
  parallel::distributed::Triangulation<dim> tria_fine(MPI_COMM_WORLD);
  create_fine_grid(tria_fine);

  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria_fine);
  dof_handler_fine.distribute_dofs(fe_fine);

  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  // setup constraint matrix
  AffineConstraints<Number> constraint_coarse(
    dof_handler_coarse.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse));
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          constraint_coarse);
  constraint_coarse.close();

  AffineConstraints<Number> constraint_fine(
    dof_handler_fine.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler_fine));
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraint_fine);
  constraint_fine.close();

  // Initialize vectors on the two levels
  LinearAlgebra::distributed::Vector<Number> u_coarse, u_fine;

  const unsigned int mg_level_fine   = numbers::invalid_unsigned_int;
  const unsigned int mg_level_coarse = numbers::invalid_unsigned_int;
  initialize_dof_vector(u_coarse, dof_handler_coarse, mg_level_coarse);
  initialize_dof_vector(u_fine, dof_handler_fine, mg_level_fine);


  // -------------------------------------------------------------


  // Start with testing the interpolation from coarse to fine. We do
  // this by interpolating a linear function onto the coarse mesh,
  // interpolating that to the fine mesh, and ensuring that the
  // difference to the original linear function is small:
  {
    u_coarse = 0;
    u_fine   = 0;

    VectorTools::interpolate(dof_handler_coarse,
                             LinearFunction<dim>(),
                             u_coarse);
    constraint_coarse.distribute(u_coarse);

    // If the polynomial degree is >=1, then the linear function is in
    // the finite element space and interpolation should result in an
    // exact representation of the original function. Test this.
    // if (fe_coarse.degree > 0){}
    //   check_zero_error(dof_handler_coarse, u_coarse);

    // Now transfer from coarse to fine mesh. This should be an
    // embedding operation and so should always result in a zero error
    VectorTools::interpolate_to_finer_mesh(
      dof_handler_coarse, u_coarse, dof_handler_fine, constraint_fine, u_fine);
    // if ((fe_coarse.degree > 0) && (fe_fine.degree > 0)){}
    //   check_zero_error(dof_handler_fine, u_fine);

  }


  // Now do it the other way around. Because we are working with a
  // linear function, even interpolating from fine to coarse should
  // result in a linear function with small error to the original
  // (infinite-dimensional) linear function.
  {
    u_coarse = 0;
    u_fine   = 0;

    VectorTools::interpolate(dof_handler_fine, LinearFunction<dim>(), u_fine);
    constraint_fine.distribute(u_fine);

    // If the polynomial degree is >=1, then the linear function is in
    // the finite element space and interpolation should result in an
    // exact representation of the original function. Test this.
    // if (fe_fine.degree > 0)
    //   check_zero_error(dof_handler_fine, u_fine);

    // Now transfer from fine to coarse mesh. This is *not* embedding
    // operation, but the result should here still be a zero error
    VectorTools::interpolate_to_coarser_mesh(dof_handler_fine,
                                             u_fine,
                                             dof_handler_coarse,
                                             constraint_coarse,
                                             u_coarse);
    // if ((fe_coarse.degree > 0) && (fe_fine.degree > 0))
    //   check_zero_error(dof_handler_coarse, u_coarse);

  }

  // Final check: If we take a random coarse vector (well, one with
  // entries 0,1,2,...), interpolate it to the finer mesh, then
  // interpolate it back, we need to be able to get the same vector as
  // before. Check this.
  {
    LinearAlgebra::distributed::Vector<Number> u_coarse_backup;
    initialize_dof_vector(u_coarse_backup, dof_handler_coarse, mg_level_coarse);

    u_coarse = 0;
    u_fine   = 0;

    for (const auto i : u_coarse.locally_owned_elements())
      u_coarse(i) = i;
    u_coarse.compress(VectorOperation::insert);

    u_coarse_backup = u_coarse;

    // Then transfer up and down:
    VectorTools::interpolate_to_finer_mesh(
      dof_handler_coarse, u_coarse, dof_handler_fine, constraint_fine, u_fine);
    VectorTools::interpolate_to_coarser_mesh(dof_handler_fine,
                                             u_fine,
                                             dof_handler_coarse,
                                             constraint_coarse,
                                             u_coarse);

    // We should have gotten the same as we started with. Check this:
    for (const auto i : u_coarse.locally_owned_elements())
      AssertThrow(std::fabs(u_coarse(i) - u_coarse_backup(i)) < 1e-12,
                  ExcInternalError());

  }
}

int main(int argc, char* argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const int fe_degree = 2;
  do_test<2,double>(FE_Q<2>(fe_degree), FE_Q<2>(fe_degree));
}