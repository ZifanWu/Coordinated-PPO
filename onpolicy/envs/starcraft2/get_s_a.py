def get_state_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           # NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat
            
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_state_enemy_feats_size()
        ally_feats_dim = self.get_state_ally_feats_size()
        own_feats_dim = self.get_state_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        if (self.use_mustalive and unit.health > 0) or (not self.use_mustalive):  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.state_pathing_grid:
                move_feats[ind: ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.state_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    if unit.health > 0:
                        enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y
                        if dist < sight_range:
                            enemy_feats[e_id, 4] = 1  # visible

                    ind = 5
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (e_unit.health / e_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (e_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type
                        ind += self.unit_type_bits

                    if self.add_center_xy:
                        enemy_feats[e_id, ind] = (e_x - center_x) / self.max_distance_x  # center X
                        enemy_feats[e_id, ind+1] = (e_y - center_y) / self.max_distance_y  # center Y

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                max_cd = self.unit_max_cooldown(al_unit)

                if al_unit.health > 0:  # visible and alive
                    if unit.health > 0:
                        if dist < sight_range:
                            ally_feats[i, 0] = 1  # visible
                        ally_feats[i, 1] = dist / sight_range  # distance
                        ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    if (self.map_type == "MMM" and al_unit.unit_type == self.medivac_id):
                        ally_feats[i, 4] = al_unit.energy / max_cd  # energy
                    else:
                        ally_feats[i, 4] = (al_unit.weapon_cooldown / max_cd)  # cooldown

                    ind = 5
                    if self.obs_all_health:
                        ally_feats[i, ind] = (al_unit.health / al_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (al_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.add_center_xy:
                        ally_feats[i, ind] = (al_x - center_x) / self.max_distance_x  # center X
                        ally_feats[i, ind+1] = (al_y - center_y) / self.max_distance_y  # center Y
                        ind += 2

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.state_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.add_center_xy:
                own_feats[ind] = (x - center_x) / self.max_distance_x  # center X
                own_feats[ind+1] = (y - center_y) / self.max_distance_y  # center Y
                ind += 2

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits

            if self.state_last_action:
                own_feats[ind:] = self.last_action[agent_id]

        state = np.concatenate((ally_feats.flatten(), 
                                enemy_feats.flatten(),
                                move_feats.flatten(),
                                own_feats.flatten()))

        # Agent id features
        if self.state_agent_id:
            agent_id_feats[agent_id] = 1.
            state = np.append(state, agent_id_feats.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return state