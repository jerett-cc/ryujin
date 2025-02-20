// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Check and illustrate the deserialization process into two vectors
// using the triangulation serialization procedure.

// This should cause an issue if we try to deserialize two vectors with the same
// triangulation.

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/logstream.h>
#include "/raid/cherry/dealii-dev/tests/tests.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <memory>

// Triangulation definitions to work wround dim=1 not having distributed triangulation.
// Copied from ryujin.
namespace
  {
    template <int dim>
    struct Proxy {
      using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    };

    template <>
    struct Proxy<1> {
      using Triangulation = dealii::parallel::shared::Triangulation<1>;
    };

  } // namespace

// Initialize a distributed vector with a value.
template <int dim, int spacedim>
void initialize_vector_with_value(const std::shared_ptr<dealii::Utilities::MPI::Partitioner> part,
				  const double val,
				  dealii::LinearAlgebra::distributed::Vector<double> &U)
{
  dealii::ConditionalOStream pout(std::cout);
  pout.set_condition(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  // Use partitioner to reinit the vector
  U.reinit(part);
  // Fill the vector with the value.
  for(unsigned int i=0; i < U.locally_owned_size(); i++)
    {
      U.local_element(i) = val+i;
      pout << U.local_element(i) << std::endl;
    }
  
}

// For the triangulation given, register_data_attach the vector U.
// Called before tr.save(...)
template <int dim, int spacedim, typename DTria>
unsigned int register_pack_vector(const dealii::LinearAlgebra::distributed::Vector<double> &U,
				  DTria &tr,
				  dealii::DoFHandler<dim> &dof_handler)
{
  const unsigned int handle = tr.register_data_attach(
	    [&](const auto cell,
	    const dealii::CellStatus status) {
	      const auto n_dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
	  //The local values in this cell.
	  std::vector<double> values(n_dofs_per_cell);
	  
	  //We must pack the doubles into a std::vector<char> for this cell.
	  std::vector<char> buffer(sizeof(double)*values.size());
	  
          const auto dof_cell = typename dealii::DoFHandler<dim>::cell_iterator(
              &cell->get_triangulation(),
              cell->level(),
              cell->index(),
              &dof_handler);

	  // Place relevant global values on this cell into the values vector. 
	  std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);
	  dof_cell->get_dof_indices(dof_indices);

	  std::transform(std::begin(dof_indices),
			 std::end(dof_indices),
			 std::begin(values),
			 [&](const auto i) {
			   return U(i);});

	  // Pack the values into the buffer
	  std::memcpy(buffer.data(), values.data(), buffer.size());

	  return buffer;
          },
        /* returns_variable_size_data =*/false);
  return handle;
}

template <int dim, int spacedim, typename DTria>
void unpack_vector(dealii::LinearAlgebra::distributed::Vector<double> &U,
		   DTria &tr,
		   unsigned int handle,
		   dealii::DoFHandler<dim> &dof_handler)
{
  tr.notify_ready_to_unpack(
       handle,
       [&](const auto &cell,
            const dealii::CellStatus status,
	   const auto &data_range) {
	 const auto n_dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
	 const std::size_t n_bytes = data_range.size();
	 std::vector<double> values(n_bytes /sizeof(double));
	 std::memcpy(values.data(),
		     &data_range[0],
		     values.size() * sizeof(double));
	 // Get the relevant indices.
	 const auto dof_cell = typename dealii::DoFHandler<dim>::cell_iterator(
              &cell->get_triangulation(),
              cell->level(),
              cell->index(),
              &dof_handler);

	  // Place relevant global values on this cell into the values vector. 
	  std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);
	  dof_cell->get_dof_indices(dof_indices);

	  for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
	    const auto global_i = dof_indices[i];
	    U(global_i) = values[i];
	  }
  
       });
}

template <int dim, int spacedim>
void
test()
{
  using DistributedTriangulation = typename Proxy<dim>::Triangulation;
  
  // Generate fulllydistributed triangulation from serial triangulation
  dealii::Triangulation<dim, spacedim> basetria;
  dealii::GridGenerator::hyper_cube(basetria);
  
  // Create distributed triangulation.
  DistributedTriangulation tr(MPI_COMM_WORLD);
  tr.copy_triangulation(basetria);
  tr.refine_global(2);

  DistributedTriangulation tr2(MPI_COMM_WORLD);
  tr2.copy_triangulation(basetria);

  // From the triangulation, create a DofHandler
  dealii::DoFHandler<dim> dof_handler(tr);
  const dealii::FE_Q<dim> fe(1);
  dof_handler.distribute_dofs(fe);

  // Generate two dealii::Distributed::Vector's with the mpi_partitioner, and fill them with
  // 'data'
  dealii::LinearAlgebra::distributed::Vector<double> u0, u1;

  const dealii::IndexSet &locally_owned = dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

  std::shared_ptr<dealii::Utilities::MPI::Partitioner> partitioner =
    std::make_shared<dealii::Utilities::MPI::Partitioner>(locally_owned,
							  locally_relevant,
							  MPI_COMM_WORLD);

  initialize_vector_with_value<dim,spacedim>(partitioner, 9.81, u0);
  initialize_vector_with_value<dim,spacedim>(partitioner, 3.14, u1);

  u0.update_ghost_values();
  u1.update_ghost_values();

  std::cout << "global size is " << u0.size() << " "  << u1.size() << std::endl;

  unsigned int handle_0 = register_pack_vector<dim, spacedim, DistributedTriangulation>(u0,
											tr,
											dof_handler);
  
  // save u0 data to archive
  std::ostringstream oss;
  {
    boost::archive::text_oarchive oa(oss, boost::archive::no_header);

    tr.save("checkpoint0");
    oa << handle_0;
    // archive and stream closed when
    // destructors are called
  }
  deallog << oss.str() << std::endl;

  
  // Now remove all information in tr and particle_handler,
  // this is like creating new objects after a restart
  //tr.clear();
  // tr.notify_ready_to_unpack
  unpack_vector<dim,spacedim,DistributedTriangulation>(u1, tr, handle_0, dof_handler);
  // verify correctness of the serialization. Note that the deserialization of
  // the particle handler has to happen before the triangulation (otherwise it
  // does not know if something was stored in the user data of the
  // triangulation).
  {
    std::istringstream            iss(oss.str());
    boost::archive::text_iarchive ia(iss, boost::archive::no_header);

    tr2.load("checkpoint0");)
    ia >> tr2;
    
  }

  // for (auto particle = particle_handler.begin();
  //      particle != particle_handler.end();
  //      ++particle)
  //   deallog << "After serialization particle id " << particle->get_id()
  //           << " is in cell " << particle->get_surrounding_cell() << std::endl;

  // deallog << "OK" << std::endl << std::endl;
}


int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPILogInitAll all;

  deallog.push("2d/2d");
  test<2, 2>();
  // deallog.pop();
  // deallog.push("2d/3d");
  // test<2, 3>();
  // deallog.pop();
  // deallog.push("3d/3d");
  // test<3, 3>();
  // deallog.pop();
}
