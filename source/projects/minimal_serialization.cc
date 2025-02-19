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


// check and illustrate the deserialization process into two vectors
// using the triangulation serialization procedure.

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

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
  for(auto &entry: U)
    pout << entry << std::endl;
  
}

template <int dim, int spacedim>
void
test()
{
  using DistributedTriangulation = typename Proxy<dim>::Triangulation;
  
  // Generate fulllydistributed triangulation from serial triangulation
  dealii::Triangulation<dim, spacedim> basetria;
  dealii::GridGenerator::hyper_cube(basetria);
  basetria.refine_global(2);

  auto construction_data =
    dealii::TriangulationDescription::Utilities::create_description_from_triangulation(
      basetria, MPI_COMM_WORLD);

  // Create distributed triangulation.
  DistributedTriangulation tr(MPI_COMM_WORLD);
  tr.create_triangulation(construction_data);

  // From the triangulation, create a DofHandler
  dealii::DoFHandler<dim> dof_handler(tr);

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

  initialize_vector_with_value<dim,spacedim>(partitioner, 0.0, u0);
  

  // // save data to archive
  // std::ostringstream oss;
  // {
  //   boost::archive::text_oarchive oa(oss, boost::archive::no_header);

  //   oa << particle_handler;
  //   tr.save("checkpoint");

  //   // archive and stream closed when
  //   // destructors are called
  // }
  // deallog << oss.str() << std::endl;

  // // Now remove all information in tr and particle_handler,
  // // this is like creating new objects after a restart
  // tr.clear();

  // // verify correctness of the serialization. Note that the deserialization of
  // // the particle handler has to happen before the triangulation (otherwise it
  // // does not know if something was stored in the user data of the
  // // triangulation).
  // {
  //   std::istringstream            iss(oss.str());
  //   boost::archive::text_iarchive ia(iss, boost::archive::no_header);

  //   ia >> particle_handler;
  //   tr.load("checkpoint");
  //   particle_handler.deserialize();
  // }

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
