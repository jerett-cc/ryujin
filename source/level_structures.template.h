/*
 * level_structures.template.h
 *
 *  Created on: Oct 5, 2023
 *      Author: jerett
 */
//todo: add comments
#include "level_structures.h"

namespace ryujin{
  namespace mgrit{

    template<typename Description, int dim, typename Number>
    LevelStructures<Description,dim, Number>::LevelStructures(const MPI_Comm& comm_x,
        const int refinement)
    : ParameterAcceptor("/LevelStructures")
    , level_comm_x(comm_x)
    , level_refinement(refinement)
    {
      discretization = std::make_shared<ryujin::Discretization<dim>>(level_comm_x,
          level_refinement);
      offline_data = std::make_shared<ryujin::OfflineData<dim, Number>>(level_comm_x,
          *discretization);

      prepare();
    }

    /**
     * this function prepares all the data structures
     */
    template<typename Description, int dim, typename Number>
    void LevelStructures<Description, dim, Number>::prepare()
    {
      discretization->prepare();
      offline_data->prepare(dim+2);
    }

  }//namespace mgrit
}//namespace ryujin


