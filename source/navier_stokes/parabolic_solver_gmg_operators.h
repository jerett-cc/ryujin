//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <offline_data.h>
#include <openmp.h>
#include <simd.h>

#include "../euler/hyperbolic_system.h"
#include "parabolic_system.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

/*
 * FIXME: generalize and make these operators equation independent and
 * refactor into ../parabolic_module_gmg_operators.h
 */

namespace ryujin
{
  namespace NavierStokes
  {
    using HyperbolicSystem = Euler::HyperbolicSystem;

    /**
     * A diagonal matrix used as a preconditioner for the non-multigrid CG
     * iteration. The diagonal matrix is constructed by computing
     * \f$d_i=1/(\rho_i m_i)\f$ for a given lumped mass matrix \f$ m_i \f$,
     * and a density field \f$ \rho_i \f$.
     *
     * @ingroup NavierStokesEquations
     */
    template <int dim, typename Number>
    class DiagonalMatrix
    {
    public:
      /**
       * @copydoc OfflineData::vector_type
       */
      using vector_type = dealii::LinearAlgebra::distributed::Vector<Number>;

      /**
       * @copydoc ParabolicModule::vector_type
       */
      using block_vector_type =
          dealii::LinearAlgebra::distributed::BlockVector<Number>;

      /**
       * Constructor
       */
      DiagonalMatrix() = default;

      /**
       * Compute the inverse of a given density \f$ \rho_i \f$ and a given
       * lumped mass matrix \f$ m_i \f$, viz., \f$d_i=1/(\rho_i m_i)\f$.
       */
      void reinit(const vector_type &lumped_mass_matrix,
                  const vector_type &density,
                  const dealii::AffineConstraints<Number> &affine_constraints)
      {
        diagonal.reinit(density, true);

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0;
             i < density.get_partitioner()->locally_owned_size();
             ++i) {
          diagonal.local_element(i) =
              Number(1.0) /
              (density.local_element(i) * lumped_mass_matrix.local_element(i));
        }

        /*
         * Fix up diagonal entries for constrained degrees of freedom due to
         * periodic boundary conditions.
         */
        affine_constraints.set_zero(diagonal);
      }

      /**
       * Get access to the internal vector to be externally filled.
       */
      vector_type &get_vector()
      {
        return diagonal;
      }

      /**
       * Get access to the internal vector to be externally filled.
       */
      block_vector_type &get_block_vector()
      {
        return diagonal_block;
      }

      /**
       * Apply on a vector.
       */
      void vmult(vector_type &dst, const vector_type &src) const
      {
        AssertDimension(diagonal_block.size(), 0);
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0;
             i < diagonal.get_partitioner()->locally_owned_size();
             ++i)
          dst.local_element(i) =
              diagonal.local_element(i) * src.local_element(i);
      }

      /**
       * Apply on a block vector.
       */
      void vmult(block_vector_type &dst, const block_vector_type &src) const
      {
        AssertDimension(dim, dst.n_blocks());
        AssertDimension(dim, src.n_blocks());
        if (diagonal_block.size() == 0) {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (unsigned int i = 0;
               i < diagonal.get_partitioner()->locally_owned_size();
               ++i)
            for (unsigned int d = 0; d < dim; ++d)
              dst.block(d).local_element(i) =
                  diagonal.local_element(i) * src.block(d).local_element(i);
        } else
          for (unsigned int d = 0; d < dim; ++d) {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = 0;
                 i < src.block(d).get_partitioner()->locally_owned_size();
                 ++i)
              dst.block(d).local_element(i) =
                  diagonal_block.block(d).local_element(i) *
                  src.block(d).local_element(i);
          }
      }

    private:
      vector_type diagonal;
      block_vector_type diagonal_block;
    };


    /**
     * An operator describing the velocity-velocity subblock of the
     * parabolic system.
     *
     * @ingroup NavierStokesEquations
     */
    template <int dim, typename Number, typename Number2>
    class VelocityMatrix : public dealii::Subscriptor
    {
    public:
      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      using vector_type = dealii::LinearAlgebra::distributed::Vector<Number>;
      using block_vector_type =
          dealii::LinearAlgebra::distributed::BlockVector<Number>;

      VelocityMatrix() = default;

      void initialize(
          const ParabolicSystem &parabolic_system,
          const OfflineData<dim, Number2> &offline_data,
          const dealii::MatrixFree<dim, Number> &matrix_free,
          const dealii::LinearAlgebra::distributed::Vector<Number> &density,
          const Number theta_x_tau,
          const unsigned int level = dealii::numbers::invalid_unsigned_int)
      {
        parabolic_system_ = &parabolic_system;
        offline_data_ = &offline_data;
        matrix_free_ = &matrix_free;
        density_ = &density;
        theta_x_tau_ = theta_x_tau;
        level_ = level;
      }

      void Tvmult(block_vector_type &dst, const block_vector_type &src) const
      {
        vmult(dst, src);
      }

      void vmult(block_vector_type &dst, const block_vector_type &src) const
      {
        /* Apply action of m_i rho_i V_i: */

        using VA = dealii::VectorizedArray<Number>;
        constexpr auto simd_length = VA::size();

        const vector_type *lumped_mass_matrix = nullptr;
        if constexpr (std::is_same<Number, Number2>::value) {
          if constexpr (std::is_same<Number, float>::value) {
            if (level_ == dealii::numbers::invalid_unsigned_int)
              lumped_mass_matrix = &offline_data_->lumped_mass_matrix();
            else
              lumped_mass_matrix =
                  &offline_data_->level_lumped_mass_matrix()[level_];
          } else {
            Assert(level_ == dealii::numbers::invalid_unsigned_int,
                   dealii::ExcInternalError());
            lumped_mass_matrix = &offline_data_->lumped_mass_matrix();
          }
        } else
          lumped_mass_matrix =
              &offline_data_->level_lumped_mass_matrix()[level_];

        const unsigned int n_owned =
            lumped_mass_matrix->get_partitioner()->locally_owned_size();
        const unsigned int size_regular = n_owned / simd_length * simd_length;

        RYUJIN_PARALLEL_REGION_BEGIN

        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < size_regular; i += simd_length) {
          const auto m_i = get_entry<VA>(*lumped_mass_matrix, i);
          const auto rho_i = get_entry<VA>(*density_, i);
          for (unsigned int d = 0; d < dim; ++d) {
            const auto temp = get_entry<VA>(src.block(d), i);
            write_entry<VA>(dst.block(d), m_i * rho_i * temp, i);
          }
        }

        RYUJIN_PARALLEL_REGION_END

        for (unsigned int i = size_regular; i < n_owned; ++i) {
          const auto m_i = lumped_mass_matrix->local_element(i);
          const auto rho_i = density_->local_element(i);

          for (unsigned int d = 0; d < dim; ++d) {
            const auto temp = src.block(d).local_element(i);
            dst.block(d).local_element(i) = m_i * rho_i * temp;
          }
        }

        /* Apply action of stress tensor: + theta * \sum_j B_ij V_j: */

        const auto integrator = [this](const auto &data,
                                       auto &dst,
                                       const auto &src,
                                       const auto range) {
          dealii::FEEvaluation<dim, order_fe, order_quad, dim, Number> velocity(
              data);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            velocity.reinit(cell);
            velocity.read_dof_values(src);
            apply_local_operator(velocity);
            velocity.distribute_local_to_global(dst);
          }
        };

        matrix_free_->template cell_loop<block_vector_type, block_vector_type>(
            integrator, dst, src, /* zero destination */ false);

        /* (5.4a) Fix up constrained degrees of freedom: */

        const auto &boundary_map =
            level_ == dealii::numbers::invalid_unsigned_int
                ? offline_data_->boundary_map()
                : offline_data_->level_boundary_map()[level_];

        for (auto entry : boundary_map) {
          // [i, normal, normal_mass, boundary_mass, id, position] = entry
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const dealii::Tensor<1, dim, Number> normal = std::get<1>(entry);
          const auto id = std::get<4>(entry);

          if (id == Boundary::slip || id == Boundary::object) {
            dealii::Tensor<1, dim, Number> V_i;
            for (unsigned int d = 0; d < dim; ++d)
              V_i[d] = dst.block(d).local_element(i);

            /* replace normal component by source */
            V_i -= 1. * (V_i * normal) * normal;
            for (unsigned int d = 0; d < dim; ++d) {
              const auto src_d = src.block(d).local_element(i);
              V_i += 1. * (src_d * normal[d]) * normal;
            }

            for (unsigned int d = 0; d < dim; ++d)
              dst.block(d).local_element(i) = V_i[d];

          } else if (id == Boundary::no_slip || id == Boundary::dirichlet) {

            /* set dst to src vector: */
            for (unsigned int d = 0; d < dim; ++d)
              dst.block(d).local_element(i) = src.block(d).local_element(i);
          }
        }
      }

      void compute_diagonal(
          std::shared_ptr<DiagonalMatrix<dim, Number>> &matrix) const
      {
        Assert(level_ != dealii::numbers::invalid_unsigned_int,
               dealii::ExcNotImplemented());
        matrix = std::make_shared<DiagonalMatrix<dim, Number>>();
        block_vector_type &vector = matrix->get_block_vector();
        vector.reinit(dim);
        for (unsigned int d = 0; d < dim; ++d)
          matrix_free_->initialize_dof_vector(vector.block(d));
        vector.collect_sizes();

        const auto &lumped_mass_matrix =
            offline_data_->level_lumped_mass_matrix()[level_];

        unsigned int dummy = 0;
        matrix_free_->template cell_loop<block_vector_type, unsigned int>(
            [this](
                const auto &data, auto &dst, const auto &, const auto range) {
              dealii::FEEvaluation<dim, order_fe, order_quad, dim, Number>
                  velocity(data);
              dealii::FEEvaluation<dim, order_fe, order_quad, dim, Number>
                  writer(data);

              for (unsigned int cell = range.first; cell < range.second;
                   ++cell) {
                velocity.reinit(cell);
                writer.reinit(cell);
                for (unsigned int i = 0; i < velocity.dofs_per_cell; ++i) {
                  for (unsigned int j = 0; j < velocity.dofs_per_cell; ++j)
                    velocity.begin_dof_values()[j] =
                        dealii::VectorizedArray<Number>();
                  velocity.begin_dof_values()[i] =
                      dealii::make_vectorized_array<Number>(1.);
                  apply_local_operator(velocity);
                  writer.begin_dof_values()[i] = velocity.begin_dof_values()[i];
                }
                writer.distribute_local_to_global(dst);
              }
            },
            vector,
            dummy,
            /* zero destination */ true);

        const unsigned int n_owned =
            lumped_mass_matrix.get_partitioner()->locally_owned_size();

        RYUJIN_PARALLEL_REGION_BEGIN

        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto m_i = lumped_mass_matrix.local_element(i);
          const auto rho_i = density_->local_element(i);
          for (unsigned int d = 0; d < dim; ++d)
            vector.block(d).local_element(i) =
                1. / (m_i * rho_i + vector.block(d).local_element(i));
        }

        RYUJIN_PARALLEL_REGION_END

        const auto &boundary_map = offline_data_->level_boundary_map()[level_];

        for (auto entry : boundary_map) {
          // [i, normal, normal_mass, boundary_mass, id, position] = entry
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const dealii::Tensor<1, dim, Number> normal = std::get<1>(entry);
          const auto id = std::get<4>(entry);

          if (id == Boundary::slip || id == Boundary::object) {
            dealii::Tensor<1, dim, Number> V_i;
            for (unsigned int d = 0; d < dim; ++d)
              V_i[d] = vector.block(d).local_element(i);

            /* replace normal component by 1. */
            V_i -= 1. * (V_i * normal) * normal;
            for (unsigned int d = 0; d < dim; ++d) {
              V_i += 1. * (1. * normal[d]) * normal;
            }

            for (unsigned int d = 0; d < dim; ++d)
              vector.block(d).local_element(i) = V_i[d];

          } else if (id == Boundary::no_slip || id == Boundary::dirichlet) {

            /* set dst to src vector: */
            for (unsigned int d = 0; d < dim; ++d)
              vector.block(d).local_element(i) = 1.;
          }
        }
      }

    private:
      const ParabolicSystem *parabolic_system_;
      const OfflineData<dim, Number2> *offline_data_;
      const dealii::MatrixFree<dim, Number> *matrix_free_;
      const vector_type *density_;
      Number theta_x_tau_;
      unsigned int level_;

      template <typename Evaluator>
      void apply_local_operator(Evaluator &velocity) const
      {
        const Number mu = parabolic_system_->mu();
        const Number lambda = parabolic_system_->lambda();

        velocity.evaluate(dealii::EvaluationFlags::gradients);

        for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
          if constexpr (dim == 1) {
            /* Workaround: no symmetric gradient for dim == 1: */
            const auto gradient = velocity.get_gradient(q);
            auto S = (4. / 3. * mu + lambda) * gradient;
            velocity.submit_gradient(theta_x_tau_ * S, q);

          } else {

            const auto symmetric_gradient = velocity.get_symmetric_gradient(q);
            const auto divergence = trace(symmetric_gradient);
            // S = (2 mu nabla^S(v) + (lambda - 2/3*mu) div(v) Id) : nabla phi
            auto S = 2. * mu * symmetric_gradient;
            for (unsigned int d = 0; d < dim; ++d)
              S[d][d] += (lambda - 2. / 3. * mu) * divergence;
            velocity.submit_symmetric_gradient(theta_x_tau_ * S, q);
          }
        }

        velocity.integrate(dealii::EvaluationFlags::gradients);
      }
    };


    /**
     * @ingroup NavierStokesEquations
     */
    template <int dim, typename Number>
    class MGTransferVelocity
        : public dealii::MGTransferBase<
              dealii::LinearAlgebra::distributed::BlockVector<Number>>
    {
    public:
      using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;
      using vector_type =
          dealii::LinearAlgebra::distributed::BlockVector<Number>;

      MGTransferVelocity() = default;

      void build(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::MGConstrainedDoFs &mg_constrained_dofs,
                 const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
                     &matrix_free)
      {
        transfer_.initialize_constraints(mg_constrained_dofs);
        transfer_.build(dof_handler);
        level_matrix_free_ = &matrix_free;
        scalar_vector.resize(matrix_free.min_level(), matrix_free.max_level());
        for (unsigned int level = matrix_free.min_level();
             level < matrix_free.max_level();
             ++level)
          matrix_free[level].initialize_dof_vector(scalar_vector[level]);
      }

      void prolongate(const unsigned int to_level,
                      vector_type &dst,
                      const vector_type &src) const override
      {
        for (unsigned int block = 0; block < src.n_blocks(); ++block)
          transfer_.prolongate(to_level, dst.block(block), src.block(block));
      }

      void restrict_and_add(const unsigned int to_level,
                            vector_type &dst,
                            const vector_type &src) const override
      {
        for (unsigned int block = 0; block < src.n_blocks(); ++block)
          transfer_.restrict_and_add(
              to_level, dst.block(block), src.block(block));
      }

      template <typename Number2>
      void interpolate_to_mg(
          const dealii::DoFHandler<dim> &dof_handler,
          dealii::MGLevelObject<scalar_type> &dst,
          const dealii::LinearAlgebra::distributed::Vector<Number2> &src) const
      {
        if (dst[dst.min_level()].size() == 0)
          for (unsigned int l = dst.min_level(); l <= dst.max_level(); ++l)
            (*level_matrix_free_)[l].initialize_dof_vector(dst[l]);
        transfer_.interpolate_to_mg(dof_handler, dst, src);
      }

      template <typename Number2>
      void
      copy_to_mg(const dealii::DoFHandler<dim> &dof_handler,
                 dealii::MGLevelObject<vector_type> &dst,
                 const dealii::LinearAlgebra::distributed::BlockVector<Number2>
                     &src) const
      {
        if (dst[dst.min_level()].size() == 0)
          for (unsigned int l = dst.min_level(); l <= dst.max_level(); ++l) {
            dst[l].reinit(src.n_blocks());
            for (unsigned int block = 0; block < src.n_blocks(); ++block)
              (*level_matrix_free_)[l].initialize_dof_vector(
                  dst[l].block(block));
            dst[l].collect_sizes();
          }

        for (unsigned int block = 0; block < src.n_blocks(); ++block) {
          transfer_.copy_to_mg(dof_handler, scalar_vector, src.block(block));
          for (unsigned int level = dst.min_level(); level <= dst.max_level();
               ++level)
            dst[level].block(block).copy_locally_owned_data_from(
                scalar_vector[level]);
        }
      }

      template <typename Number2>
      void copy_from_mg(
          const dealii::DoFHandler<dim> &dof_handler,
          dealii::LinearAlgebra::distributed::BlockVector<Number2> &dst,
          const dealii::MGLevelObject<vector_type> &src) const
      {
        for (unsigned int block = 0; block < dst.n_blocks(); ++block) {
          for (unsigned int level = src.min_level(); level <= src.max_level();
               ++level)
            scalar_vector[level].copy_locally_owned_data_from(
                src[level].block(block));
          transfer_.copy_from_mg(dof_handler, dst.block(block), scalar_vector);
        }
      }

    private:
      dealii::MGTransferMatrixFree<dim, Number> transfer_;
      const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
          *level_matrix_free_;
      mutable dealii::MGLevelObject<scalar_type> scalar_vector;
    };


    /**
     * @ingroup NavierStokesEquations
     */
    template <int dim, typename Number, typename Number2>
    class EnergyMatrix : public dealii::Subscriptor
    {
    public:
      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      using vector_type = dealii::LinearAlgebra::distributed::Vector<Number>;

      EnergyMatrix() = default;

      void initialize(
          const OfflineData<dim, Number2> &offline_data,
          const dealii::MatrixFree<dim, Number> &matrix_free,
          const dealii::LinearAlgebra::distributed::Vector<Number> &density,
          const Number time_factor,
          const unsigned int level = dealii::numbers::invalid_unsigned_int)
      {
        offline_data_ = &offline_data;
        matrix_free_ = &matrix_free;
        density_ = &density;
        factor_ = time_factor;
        level_ = level;
      }

      void Tvmult(vector_type &dst, const vector_type &src) const
      {
        vmult(dst, src);
      }

      dealii::types::global_dof_index m() const
      {
        return density_->size();
      }

      Number el(const unsigned int, const unsigned int) const
      {
        Assert(false, dealii::ExcNotImplemented());
        return Number();
      }

      void vmult(vector_type &dst, const vector_type &src) const
      {
        /* Apply action of m_i rho_i V_i: */

        using VA = dealii::VectorizedArray<Number>;
        constexpr auto simd_length = VA::size();

        const vector_type *lumped_mass_matrix = nullptr;
        if constexpr (std::is_same<Number, Number2>::value) {
          if constexpr (std::is_same<Number, float>::value) {
            if (level_ == dealii::numbers::invalid_unsigned_int)
              lumped_mass_matrix = &offline_data_->lumped_mass_matrix();
            else
              lumped_mass_matrix =
                  &offline_data_->level_lumped_mass_matrix()[level_];
          } else {
            Assert(level_ == dealii::numbers::invalid_unsigned_int,
                   dealii::ExcInternalError());
            lumped_mass_matrix = &offline_data_->lumped_mass_matrix();
          }
        } else
          lumped_mass_matrix =
              &offline_data_->level_lumped_mass_matrix()[level_];

        const unsigned int n_owned =
            lumped_mass_matrix->get_partitioner()->locally_owned_size();
        const unsigned int size_regular = n_owned / simd_length * simd_length;

        RYUJIN_PARALLEL_REGION_BEGIN

        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < size_regular; i += simd_length) {
          const auto m_i = get_entry<VA>(*lumped_mass_matrix, i);
          const auto rho_i = get_entry<VA>(*density_, i);
          const auto e_i = get_entry<VA>(src, i);
          write_entry<VA>(dst, m_i * rho_i * e_i, i);
        }

        RYUJIN_PARALLEL_REGION_END

        for (unsigned int i = size_regular; i < n_owned; ++i) {
          const auto m_i = lumped_mass_matrix->local_element(i);
          const auto rho_i = density_->local_element(i);
          const auto e_i = src.local_element(i);
          dst.local_element(i) = m_i * rho_i * e_i;
        }

        /* Apply action of diffusion operator \sum_j beta_ij e_j: */

        const auto integrator = [this](const auto &data,
                                       auto &dst,
                                       const auto &src,
                                       const auto range) {
          dealii::FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(
              data);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            energy.reinit(cell);
            energy.read_dof_values(src);
            apply_local_operator(energy);
            energy.distribute_local_to_global(dst);
          }
        };

        matrix_free_->template cell_loop<vector_type, vector_type>(
            integrator, dst, src, /* zero destination */ false);

        /* Fix up constrained degrees of freedom: */

        const auto &boundary_map =
            (level_ == dealii::numbers::invalid_unsigned_int)
                ? offline_data_->boundary_map()
                : offline_data_->level_boundary_map()[level_];

        for (auto entry : boundary_map) {
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const auto id = std::get<4>(entry);
          if (id == Boundary::dirichlet)
            dst.local_element(i) = src.local_element(i);
        }
      }

      void compute_diagonal(
          std::shared_ptr<dealii::DiagonalMatrix<vector_type>> &matrix) const
      {
        Assert(level_ != dealii::numbers::invalid_unsigned_int,
               dealii::ExcNotImplemented());
        matrix = std::make_shared<dealii::DiagonalMatrix<vector_type>>();
        vector_type &vector = matrix->get_vector();
        matrix_free_->initialize_dof_vector(vector);

        const vector_type &lumped_mass_matrix =
            offline_data_->level_lumped_mass_matrix()[level_];

        unsigned int dummy = 0;
        matrix_free_->template cell_loop<vector_type, unsigned int>(
            [this](
                const auto &data, auto &dst, const auto &, const auto range) {
              dealii::FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(
                  data);
              dealii::FEEvaluation<dim, order_fe, order_quad, 1, Number> writer(
                  data);

              for (unsigned int cell = range.first; cell < range.second;
                   ++cell) {
                energy.reinit(cell);
                writer.reinit(cell);
                for (unsigned int i = 0; i < energy.dofs_per_cell; ++i) {
                  for (unsigned int j = 0; j < energy.dofs_per_cell; ++j)
                    energy.begin_dof_values()[j] =
                        dealii::VectorizedArray<Number>();
                  energy.begin_dof_values()[i] =
                      dealii::make_vectorized_array<Number>(1.);
                  apply_local_operator(energy);
                  writer.begin_dof_values()[i] = energy.begin_dof_values()[i];
                }
                writer.distribute_local_to_global(dst);
              }
            },
            vector,
            dummy,
            /* zero destination */ true);

        const unsigned int n_owned =
            lumped_mass_matrix.get_partitioner()->locally_owned_size();

        RYUJIN_PARALLEL_REGION_BEGIN

        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto m_i = lumped_mass_matrix.local_element(i);
          const auto rho_i = density_->local_element(i);
          vector.local_element(i) =
              1. / (m_i * rho_i + vector.local_element(i));
        }

        RYUJIN_PARALLEL_REGION_END

        const auto &boundary_map = offline_data_->level_boundary_map()[level_];

        for (auto entry : boundary_map) {
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const auto id = std::get<4>(entry);
          if (id == Boundary::dirichlet)
            vector.local_element(i) = 1.;
        }
      }

    private:
      const OfflineData<dim, Number2> *offline_data_;
      const dealii::MatrixFree<dim, Number> *matrix_free_;
      const dealii::LinearAlgebra::distributed::Vector<Number> *density_;
      Number factor_;
      unsigned int level_;

      template <typename Evaluator>
      void apply_local_operator(Evaluator &energy) const
      {
        energy.evaluate(dealii::EvaluationFlags::gradients);
        for (unsigned int q = 0; q < energy.n_q_points; ++q) {
          energy.submit_gradient(factor_ * energy.get_gradient(q), q);
        }
        energy.integrate(dealii::EvaluationFlags::gradients);
      }
    };


    /**
     * @ingroup NavierStokesEquations
     */
    template <int dim, typename Number>
    class MGTransferEnergy : public dealii::MGTransferMatrixFree<dim, Number>
    {
    public:
      void build(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
                     &matrix_free)
      {
        dealii::MGTransferMatrixFree<dim, Number>::build(dof_handler);
        level_matrix_free_ = &matrix_free;
      }

      template <typename Number2>
      void copy_to_mg(
          const dealii::DoFHandler<dim> &dof_handler,
          dealii::MGLevelObject<
              dealii::LinearAlgebra::distributed::Vector<Number>> &dst,
          const dealii::LinearAlgebra::distributed::Vector<Number2> &src) const
      {
        if (dst[dst.min_level()].size() == 0)
          for (unsigned int l = dst.min_level(); l <= dst.max_level(); ++l)
            (*level_matrix_free_)[l].initialize_dof_vector(dst[l]);
        dealii::MGTransferMatrixFree<dim, Number>::copy_to_mg(
            dof_handler, dst, src);
      }

    private:
      const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
          *level_matrix_free_;
    };

  } // namespace NavierStokes
} /* namespace ryujin */

#undef locally_owned_size
