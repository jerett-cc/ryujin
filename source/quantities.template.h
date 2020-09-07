//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef QUANTITIES_TEMPLATE_H
#define QUANTITIES_TEMPLATE_H

#include "quantities.h"
#include "simd.h"

#include <fstream>

namespace ryujin
{
  using namespace dealii;

  template <int dim, typename Number>
  Quantities<dim, Number>::Quantities(
      const MPI_Comm &mpi_communicator,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , mpi_rank(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , offline_data_(&offline_data)
  {
    compute_conserved_quantities_ = true;
    add_parameter("compute conserved quantities",
                  compute_conserved_quantities_,
                  "Compute and write the conserved quantities to a logfile at "
                  "specified intervals");
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::prepare(const std::string &name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif
    if (mpi_rank != 0)
      return;

    output.open(name);

    output << "time";
    if (compute_conserved_quantities_)
      std::cout << "\ttotal mass\ttotal momentum\ttotal energy";
    output << std::endl;
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::compute(const vector_type &U, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::compute()" << std::endl;
#endif
    if (mpi_rank == 0)
      output << std::scientific << std::setprecision(14) << t;

    if (compute_conserved_quantities_) {
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();

      rank1_type summed_quantities;

      RYUJIN_PARALLEL_REGION_BEGIN

      rank1_type summed_quantities_thread_local;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_owned; ++i) {
        const auto m_i = lumped_mass_matrix.local_element(i);
        const auto U_i = U.get_tensor(i);
        summed_quantities_thread_local += m_i * U_i;
      }

      RYUJIN_OMP_CRITICAL
      summed_quantities += summed_quantities_thread_local;

      RYUJIN_PARALLEL_REGION_END

      for (unsigned int k = 0; k < problem_dimension; ++k) {
        summed_quantities[k] =
            Utilities::MPI::sum(summed_quantities[k], mpi_communicator_);
        if (mpi_rank == 0)
          output << "\t" << summed_quantities[k];
      }
    }

    if (mpi_rank == 0)
      output << std::endl;
  }

} /* namespace ryujin */

#endif /* QUANTITIES_TEMPLATE_H */
