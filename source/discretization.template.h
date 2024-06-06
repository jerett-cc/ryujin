//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "geometry_library.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <random>

namespace ryujin
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const MPI_Comm &mpi_communicator,
                                      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
  {
    const auto smoothing =
        dealii::Triangulation<dim>::limit_level_difference_at_vertices;

    if constexpr (have_distributed_triangulation<dim>) {
      const auto settings =
          Triangulation::Settings::construct_multigrid_hierarchy;
      triangulation_ = std::make_unique<Triangulation>(
          mpi_communicator_, smoothing, settings);

    } else {
      const auto settings = static_cast<typename Triangulation::Settings>(
          Triangulation::partition_auto |
          Triangulation::construct_multigrid_hierarchy);
      /* Beware of the boolean: */
      triangulation_ = std::make_unique<Triangulation>(
          mpi_communicator_, smoothing, /*artificial cells*/ true, settings);
    }

    /* Options: */

    ansatz_ = Ansatz::cg_q1;
    add_parameter("finite element ansatz",
                  ansatz_,
                  "The finite element ansatz used for discretization. valid "
                  "choices are cG Q1, cG Q2, cG Q3.");

    geometry_ = "cylinder";
    add_parameter("geometry",
                  geometry_,
                  "Name of the geometry used to create the mesh. Valid names "
                  "are given by any of the subsections defined below.");

    refinement_ = 5;
    add_parameter("mesh refinement",
                  refinement_,
                  "number of refinement of global refinement steps");

    mesh_distortion_ = 0.;
    add_parameter(
        "mesh distortion", mesh_distortion_, "Strength of mesh distortion");

    repartitioning_ = false;
    add_parameter("mesh repartitioning",
                  repartitioning_,
                  "try to equalize workload by repartitioning the mesh");

    Geometries::populate_geometry_list<dim>(geometry_list_, subsection);
  }

  /**
   * This constructor takes a @p mpi_communicator and @p refinement and
   * sets the refinement to be the user specified one, rather than the one
   * the input file specifies.
   */
  template <int dim>
   Discretization<dim>::Discretization(const MPI_Comm &mpi_communicator,
                                       const unsigned int refinement,
                                       const std::string &subsection)
       : Discretization(mpi_communicator, subsection)
   {
     refinement_ = refinement;
   }


  template <int dim>
  void Discretization<dim>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Discretization<dim>::prepare()" << std::endl;
#endif

    auto &triangulation = *triangulation_;
    triangulation.clear();

    {
      bool initialized = false;
      for (auto &it : geometry_list_)
        if (it->name() == geometry_) {
          it->create_triangulation(triangulation);
          initialized = true;
          break;
        }

      AssertThrow(
          initialized,
          ExcMessage("Could not find a geometry description with name \"" +
                     geometry_ + "\""));
    }

    if constexpr (have_distributed_triangulation<dim>) {
      if (repartitioning_) {
        /*
         * Try to partition the mesh equilibrating the workload. The usual mesh
         * partitioning heuristic that tries to partition the mesh such that
         * every MPI rank has roughly the same number of locally owned degrees
         * of freedom does not work well in our case due to the fact that
         * boundary dofs are not SIMD parallelized. (In fact, every dof with
         * "non-standard connectivity" is not SIMD parallelized. Those are
         * however exceedingly rare (point irregularities in 2D, line
         * irregularities in 3D) and we simply ignore them.)
         *
         * For the mesh partitioning scheme we have to supply an additional
         * weight that gets added to the default weight of a cell which is
         * 1000. Asymptotically we have one boundary dof per boundary cell (in
         * any dimension). A rough benchmark reveals that the speedup due to
         * SIMD vectorization is typically less than VectorizedArray::size() /
         * 2. Boundary dofs are more expensive due to certain special treatment
         * (additional symmetrization of d_ij, boundary fixup) so it should be
         * safe to assume that the cost incurred is at least
         * VectorizedArray::size() / 2.
         */
        constexpr auto speedup = dealii::VectorizedArray<NUMBER>::size() / 2u;
        constexpr unsigned int weight = 1000u;

#if DEAL_II_VERSION_GTE(9, 5, 0)
        triangulation.signals.weight.connect(
#else
        triangulation.signals.cell_weight.connect(
#endif
            [](const auto &cell, const auto /*status*/) -> unsigned int {
              if (cell->at_boundary())
                return weight * (speedup == 0u ? 0u : speedup - 1u);
              else
                return 0u;
            });

        triangulation.repartition();
      }
    }

    triangulation.refine_global(refinement_);

    if (std::abs(mesh_distortion_) > 1.0e-10)
      GridTools::distort_random(
          mesh_distortion_, triangulation, std::random_device()());

    switch (ansatz_) {
    case Ansatz::cg_q1:
      finite_element_ = std::make_unique<FE_Q<dim>>(1);
      break;
    case Ansatz::cg_q2:
      finite_element_ = std::make_unique<FE_Q<dim>>(2);
      break;
    case Ansatz::cg_q3:
      finite_element_ = std::make_unique<FE_Q<dim>>(3);
      break;
    case Ansatz::dg_q0:
      finite_element_ = std::make_unique<FE_DGQ<dim>>(0);
      break;
    case Ansatz::dg_q1:
      finite_element_ = std::make_unique<FE_DGQ<dim>>(1);
      break;
    case Ansatz::dg_q2:
      finite_element_ = std::make_unique<FE_DGQ<dim>>(2);
      break;
    case Ansatz::dg_q3:
      finite_element_ = std::make_unique<FE_DGQ<dim>>(3);
      break;
    }

    switch (ansatz_) {
    case Ansatz::dg_q0:
      mapping_ = std::make_unique<MappingQ<dim>>(1);
      quadrature_ = std::make_unique<QGauss<dim>>(1);
      quadrature_1d_ = std::make_unique<QGauss<1>>(1);
      face_quadrature_ = std::make_unique<QGauss<dim - 1>>(1);
      if (dim > 1)
        face_nodal_quadrature_ = std::make_unique<dealii::QMidpoint<dim - 1>>();
      else
        face_nodal_quadrature_ = std::make_unique<QGauss<dim - 1>>(1);
      break;
    case Ansatz::cg_q1:
      [[fallthrough]];
    case Ansatz::dg_q1:
      mapping_ = std::make_unique<MappingQ<dim>>(1);
      quadrature_ = std::make_unique<QGauss<dim>>(2);
      quadrature_1d_ = std::make_unique<QGauss<1>>(2);
      face_quadrature_ = std::make_unique<QGauss<dim - 1>>(2);
      face_nodal_quadrature_ =
          std::make_unique<dealii::QGaussLobatto<dim - 1>>(2);
      break;
    case Ansatz::cg_q2:
      [[fallthrough]];
    case Ansatz::dg_q2:
      mapping_ = std::make_unique<MappingQ<dim>>(2);
      quadrature_ = std::make_unique<QGauss<dim>>(3);
      quadrature_1d_ = std::make_unique<QGauss<1>>(3);
      face_quadrature_ = std::make_unique<QGauss<dim - 1>>(3);
      face_nodal_quadrature_ =
          std::make_unique<dealii::QGaussLobatto<dim - 1>>(3);
      break;
    case Ansatz::cg_q3:
      [[fallthrough]];
    case Ansatz::dg_q3:
      mapping_ = std::make_unique<MappingQ<dim>>(3);
      quadrature_ = std::make_unique<QGauss<dim>>(4);
      quadrature_1d_ = std::make_unique<QGauss<1>>(4);
      face_quadrature_ = std::make_unique<QGauss<dim - 1>>(4);
      face_nodal_quadrature_ =
          std::make_unique<dealii::QGaussLobatto<dim - 1>>(4);
      break;
    }
  }

} /* namespace ryujin */
