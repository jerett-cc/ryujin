//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "quantities.h"
#include "scope.h"
#include "scratch_data.h"
#include "simd.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN
template <int rank, int dim, typename Number>
bool operator<(const Tensor<rank, dim, Number> &left,
               const Tensor<rank, dim, Number> &right)
{
  return std::lexicographical_compare(
      left.begin_raw(), left.end_raw(), right.begin_raw(), right.end_raw());
}
DEAL_II_NAMESPACE_CLOSE

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  Quantities<dim, Number>::Quantities(
      const MPI_Comm &mpi_communicator,
      const ryujin::ProblemDescription &problem_description,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , problem_description_(&problem_description)
      , offline_data_(&offline_data)
  {
    add_parameter("interior manifolds",
                  interior_manifolds_,
                  "List of level set functions describing interior manifolds. "
                  "The description is used to only output point values for "
                  "vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");

    boundary_manifolds_.push_back(
        {"upper_boundary", "y - 1.0", "time_averaged instantaneous"});
    boundary_manifolds_.push_back(
        {"lower_boundary", "y + 1.0", "time_averaged instantaneous"});
    add_parameter("boundary manifolds",
                  boundary_manifolds_,
                  "List of level set functions describing boundary. The "
                  "description is used to only output point values for "
                  "boundary vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::prepare(std::string name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;

    // FIXME
    AssertThrow(interior_manifolds_.empty(),
                dealii::ExcMessage("Not implemented. Output for interior "
                                   "manifolds has not been written yet."));

    const unsigned int n_owned = offline_data_->n_locally_owned();

    /*
     * Create boundary maps and allocate statistics vector:
     */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::back_inserter(boundary_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression, option] = it;
          FunctionParser<dim> level_set_function(expression);

          std::vector<boundary_point> map;

          for (const auto &entry : offline_data_->boundary_map()) {
            /* skip nonlocal */
            if (entry.first >= n_owned)
              continue;

            /* skip constrained */
            if (offline_data_->affine_constraints().is_constrained(
                    offline_data_->scalar_partitioner()->local_to_global(
                        entry.first)))
              continue;

            const auto &[normal, normal_mass, boundary_mass, id, position] =
                entry.second;
            if (std::abs(level_set_function.value(position)) < 1.e-12)
              map.push_back({entry.first,
                             normal,
                             normal_mass,
                             boundary_mass,
                             id,
                             position});
          }
          return std::make_tuple(name, map);
        });

    boundary_statistics_.clear();
    boundary_statistics_.resize(boundary_manifolds_.size());

    for (std::size_t i = 0; i < boundary_maps_.size(); ++i) {
      const auto n_entries = std::get<1>(boundary_maps_[i]).size();
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[i];
      val_old.resize(n_entries);
      val_new.resize(n_entries);
      val_sum.resize(n_entries);
      t_old = t_new = t_sum = 0.;
    }

    boundary_time_series_.clear();
    boundary_time_series_.resize(boundary_manifolds_.size());

    /*
     * Output boundary maps:
     */

    for (const auto &it : boundary_maps_) {
      const auto &[description, boundary_map] = it;

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received =
          Utilities::MPI::gather(mpi_communicator_, boundary_map);

      if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

        std::ofstream output(base_name_ + "-" + description + "-points.dat");
        output << std::scientific << std::setprecision(14);

        output << "#\n# position\tnormal\tnormal mass\tboundary mass\n";

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << "\n";
          for (const auto &entry : entries) {
            const auto &[index, n_i, nm_i, bm_i, id, x_i] = entry;
            output << x_i << "\t" << n_i << "\t" << nm_i << "\t" << bm_i
                   << "\n";
          } /*entry*/
        }   /*entries*/

        output << std::flush;
      }
    }
  }


  template <int dim, typename Number>
  auto Quantities<dim, Number>::accumulate_internal(
      const vector_type &U,
      const std::vector<boundary_point> &boundary_map,
      std::vector<boundary_value> &val_new) -> boundary_value
  {
    boundary_value spatial_average;
    Number nm_sum = Number(0.);
    Number bm_sum = Number(0.);

    std::transform(
        boundary_map.begin(),
        boundary_map.end(),
        val_new.begin(),
        [&](auto point) -> boundary_value {
          const auto &[i, n_i, nm_i, bm_i, id, x_i] = point;

          const auto U_i = U.get_tensor(i);
          const auto rho_i = problem_description_->density(U_i);
          const auto m_i = problem_description_->momentum(U_i);
          const auto v_i = m_i / rho_i;
          const auto p_i = problem_description_->pressure(U_i);

          // FIXME: compute symmetric diffusion tensor s(U) over stencil
          const auto S_i = dealii::Tensor<2, dim, Number>();
          const auto tau_n_i = S_i * n_i;

          boundary_value result;

          if constexpr (dim == 1)
            result = {state_type{{rho_i, v_i[0], p_i}},
                      state_type{{rho_i * rho_i, v_i[0] * v_i[0], p_i * p_i}},
                      tau_n_i,
                      p_i * n_i};
          else if constexpr (dim == 2)
            result = {state_type{{rho_i, v_i[0], v_i[1], p_i}},
                      state_type{{rho_i * rho_i,
                                  v_i[0] * v_i[0],
                                  v_i[1] * v_i[1],
                                  p_i * p_i}},
                      tau_n_i,
                      p_i * n_i};
          else
            result = {state_type{{rho_i, v_i[0], v_i[1], v_i[2], p_i}},
                      state_type{{rho_i * rho_i,
                                  v_i[0] * v_i[0],
                                  v_i[1] * v_i[1],
                                  v_i[2] * v_i[2],
                                  p_i * p_i}},
                      tau_n_i,
                      p_i * n_i};

          bm_sum += bm_i;
          std::get<0>(spatial_average) += bm_i * std::get<0>(result);
          std::get<1>(spatial_average) += bm_i * std::get<1>(result);

          nm_sum += nm_i;
          std::get<2>(spatial_average) += nm_i * std::get<2>(result);
          std::get<3>(spatial_average) += nm_i * std::get<3>(result);

          return result;
        });

      std::get<0>(spatial_average) /= bm_sum;
      std::get<1>(spatial_average) /= bm_sum;
      std::get<2>(spatial_average) /= nm_sum;
      std::get<3>(spatial_average) /= nm_sum;

      return spatial_average;
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out_internal(
      std::ostream &output,
      const std::vector<boundary_value> &values,
      const Number scale)
  {
    /*
     * FIXME: This currently distributes boundary maps to all MPI ranks.
     * This is unnecessarily wasteful. Ideally, we should do MPI IO with
     * only MPI ranks participating who actually have boundary values.
     */

    const auto received =
        Utilities::MPI::gather(mpi_communicator_, values);

    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      output << "# primitive state (rho, u, p)\t2nd moments (rho^2, u_i^2, "
                "p^2)\tboundary stress\tpressure normal\n";

      unsigned int rank = 0;
      for (const auto &entries : received) {
        output << "# rank " << rank++ << "\n";
        for (const auto &entry : entries) {
          const auto &[state, state_square, tau_n, pn] = entry;
          output << scale * state << "\t" << scale * state_square << "\t"
                 << scale * tau_n << "\t" << scale * pn << "\n";
        } /*entry*/
      }   /*entries*/

      output << std::flush;
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out_time_series(
      std::ostream &output,
      const std::vector<std::tuple<Number, boundary_value>> &values,
      bool append)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator_) == 0) {

      if (!append)
        output << "# time t\tprimitive state (rho, u, p)\t2nd moments (rho^2, "
                  "u_i^2, p^2)\tboundary stress\tpressure normal\n";

      for (const auto &entry : values) {
        const auto t = std::get<0>(entry);
        const auto &[state, state_square, tau_n, pn] = std::get<1>(entry);

        output << t << "\t" << state << "\t" << state_square << "\t" << tau_n
               << "\t" << pn << "\n";
      }

      output << std::flush;
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::accumulate(const vector_type &U, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    for (std::size_t i = 0; i < boundary_maps_.size(); ++i) {

      const auto options = std::get<2>(boundary_manifolds_[i]);

      /* skip if we don't average in space or time: */
      if (options.find("time_averaged") == std::string::npos &&
          options.find("space_averaged") == std::string::npos)
        continue;

      const auto boundary_map = std::get<1>(boundary_maps_[i]);
      auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
          boundary_statistics_[i];

      std::swap(t_old, t_new);
      std::swap(val_old, val_new);

      /* accumulate new values */

      const auto spatial_average = accumulate_internal(U, boundary_map, val_new);

      /* Average in time with trapezoidal rule: */

      if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
        /* We have not accumulated any statistics yet: */
        t_old = t - 1.;
        t_new = t;

      } else {

        t_new = t;
        const Number tau = t_new - t_old;

        for (std::size_t i = 0; i < val_sum.size(); ++i) {
          /* sometimes I miss haskell's type classes... */
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
          std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
          std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
          std::get<2>(val_sum[i]) += 0.5 * tau * std::get<2>(val_old[i]);
          std::get<2>(val_sum[i]) += 0.5 * tau * std::get<2>(val_new[i]);
          std::get<3>(val_sum[i]) += 0.5 * tau * std::get<3>(val_old[i]);
          std::get<3>(val_sum[i]) += 0.5 * tau * std::get<3>(val_new[i]);
        }
        t_sum += tau;
      }

      /* Record average in space: */
      boundary_time_series_[i].push_back({t, spatial_average});
    }
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::write_out(const vector_type &U,
                                          const Number t,
                                          unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::write_out()" << std::endl;
#endif

    for (std::size_t i = 0; i < boundary_maps_.size(); ++i) {

      const auto description = std::get<0>(boundary_manifolds_[i]);
      const auto options = std::get<2>(boundary_manifolds_[i]);

      if (options.find("instantaneous") != std::string::npos) {
        /* Compute and output instantaneous field: */

        const auto boundary_map = std::get<1>(boundary_maps_[i]);
        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            boundary_statistics_[i];

        /* We have not computed any updated statistics yet: */

        if (options.find("time_averaged") == std::string::npos &&
            options.find("space_averaged") == std::string::npos)
          accumulate_internal(U, boundary_map, val_new);
        else
          AssertThrow(t_new == t, dealii::ExcInternalError());

        std::string file_name = base_name_ + "-" + description +
                                "-instantaneous-" +
                                Utilities::to_string(cycle, 6) + ".dat";
        std::ofstream output(file_name);

        output << std::scientific << std::setprecision(14);
        output << "# at t = " << t << std::endl;

        write_out_internal(output, val_new, Number(1.));
      }

      if (options.find("time_averaged") != std::string::npos) {
        /* Output time averaged field: */

        std::string file_name =
            base_name_ + "-" + description + "-time_averaged.dat";
        std::ofstream output(file_name);

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            boundary_statistics_[i];

        output << std::scientific << std::setprecision(14);
        output << "# averaged from t = " << t_new - t_sum << " to t = " << t_new
               << std::endl;

        write_out_internal(output, val_sum, Number(1.) / t_sum);
      }

      if (options.find("space_averaged") != std::string::npos) {
        /* Output space averaged field: */

        auto &time_series = boundary_time_series_[i];

        std::string file_name =
            base_name_ + "-" + description + "-space_averaged_time_series.dat";

        std::ofstream output;
        output << std::scientific << std::setprecision(14);

        if (cycle == 0) {
          output.open(file_name, std::ofstream::out | std::ofstream::trunc);
          write_out_time_series(output, time_series, /*append*/ false);

        } else {

          output.open(file_name, std::ofstream::out | std::ofstream::app);
          write_out_time_series(output, time_series, /*append*/ true);
        }

        time_series.clear();
      }
    } /* i */
  }

} /* namespace ryujin */
