import numpy as np


"""
 Function for processing game info and such
"""

def process_state(game_state, view_distance, agentIndex):
        """ (x,y) are positions on a Pacman map with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner."""
        food = np.array(game_state.data.food.data)
        walls = np.array(game_state.data.layout.walls.data)
        map_shape = walls.shape
        capsules = game_state.data.capsules
        pacman_pos = game_state.data.agentStates[0].configuration.pos   
        gosts_pos = list(map(lambda agent: agent.configuration.pos,
                             game_state.data.agentStates[1:]))
        gosts_scared = list(
            map(lambda agent: agent.scaredTimer > 0, game_state.data.agentStates[1:]))

        agent_pos = game_state.data.agentStates[agentIndex].configuration.pos
        agent_pos = int(agent_pos[0]), int(agent_pos[1])
        """
            0: empty,
            1: wall,
            2: food,
            3: capsules,
            4: ghost,
            5: scared ghost,
            6: pacman
            7: playing ghost
            8: playing scared ghost
        """

        view_slices = ((max(agent_pos[0]-view_distance[0], 0), min(agent_pos[0]+view_distance[0]+1, map_shape[0])),
                       (max(agent_pos[1]-view_distance[1], 0), min(agent_pos[1]+view_distance[1]+1, map_shape[1])))
        view_slices = ((int(view_slices[0][0]), int(view_slices[0][1])),(int(view_slices[1][0]), int(view_slices[1][1])))
        def select(l):
            return l[view_slices[0][0]:view_slices[0][1], view_slices[1][0]:view_slices[1][1]]

        obs = np.vectorize(lambda v: 1 if v else 0)(select(walls))
        obs = obs + np.vectorize(lambda v: 2 if v else 0)(select(food))

        def pos_to_relative_pos(pos):
            if (pos[0] < view_slices[0][0] or view_slices[0][1] <= pos[0]
                    or pos[1] < view_slices[1][0] or view_slices[1][1] <= pos[1]):
                return None
            else:
                return int(pos[0]-view_slices[0][0]), int(pos[1]-view_slices[1][0])

        for c_relative_pos in filter(lambda x: x is not None, map(pos_to_relative_pos, capsules)):
            obs[c_relative_pos[0], c_relative_pos[1]] = 3

        for i, g_relative_pos in enumerate(map(pos_to_relative_pos, gosts_pos)):
            if (g_relative_pos is not None):
                obs[int(g_relative_pos[0]), int(g_relative_pos[1])
                    ] = 5 if gosts_scared[i] else 4

        pacman_relative_pos = pos_to_relative_pos(pacman_pos)
        if pacman_relative_pos is not None:
            obs[pacman_relative_pos[0], pacman_relative_pos[1]] = 6

        agent_relative_pos = pos_to_relative_pos(agent_pos)
        if agentIndex != 0:
            obs[agent_relative_pos[0], agent_relative_pos[1]] = 8 if gosts_scared[agentIndex-1] else 7

        obs[0, 0] = 2 if np.any(
            food[0:agent_pos[0]+1, 0:agent_pos[1]+1]) else 0
        obs[obs.shape[0]-1,
            0] = 2 if np.any(food[agent_pos[0]:map_shape[0], 0:agent_pos[1]+1])else 0

        obs[0, obs.shape[1] -
            1] = 2 if np.any(food[0:agent_pos[0]+1, agent_pos[1]:map_shape[0]]) else 0
        obs[obs.shape[0]-1, obs.shape[1]-1] = 2 if np.any(
            food[agent_pos[0]:map_shape[0], agent_pos[1]:map_shape[0]]) else 0

        # print()
        obs = np.transpose(obs)[::-1, :]
        return obs