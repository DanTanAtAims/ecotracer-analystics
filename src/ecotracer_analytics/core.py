import numpy as np
from pyewe import CoreInterface

def get_detritus_out(core: CoreInterface):
    c_core = core._core
    ecosim_data = c_core.get_EcosimDataStructures()
    detritus_out = np.array(ecosim_data.DetritusOut)

    return detritus_out[1:] # drop unused 0-index

class EcopathSource:

    def __init__(self, core: CoreInterface):
        # We do not store core to ensure picklability
        
        self.consumption = np.array(core.Ecopath.get_consumption())[:, 1:] # prey x predator
        self.biomass = np.array(core.Ecopath.get_biomass())
        self.immigration_rate = np.array(core.Ecopath.get_immigration())
        self.emigration_rate = np.array(core.Ecopath.get_emigration())

        self.fishing_mort_rate = np.array(core.Ecopath.get_fishing_mortality_coefficient())
        self.predation_mort_rate = np.array(core.Ecopath.get_predation_mortality_coefficient())
        self.other_mort_rate = np.array(core.Ecopath.get_other_mort_coefficient())
        self.discard_fate = np.array(core.Ecopath.get_discard_fate())
        self.discard_mortality = np.array(core.Ecopath.get_discard_mortality())
        self.discards = np.array(core.Ecopath.get_discards())

        det_mask = np.array(core.Ecopath.get_detritus_mask(), dtype=bool)
        self.is_detritus = np.where(det_mask)[0]
        self.not_detritus = np.where(~det_mask)[0]

        self.det_immig = self.immigration_rate[self.is_detritus]
        self.det_ba = np.array(core.Ecopath.get_biomass_accumulation_output())[self.is_detritus]
        self.det_import = np.array(core.Ecopath.get_detritus_import())[self.is_detritus]
        self.det_fate = np.array(core.Ecopath.get_detritus_fate())[self.not_detritus]
        self.other_mort = self.other_mort_rate[self.not_detritus]
        self.detritus_out_rates = get_detritus_out(core)

    def shapes(self):
        print(f"consumption: {self.consumption.shape}")
        print(f"biomass: {self.biomass.shape}")
        print(f"immigration_rate: {self.immigration_rate.shape}")
        print(f"emigration_rate: {self.emigration_rate.shape}")

        print(f"fishing_mort_rate: {self.fishing_mort_rate.shape}")
        print(f"predation_mort_rate: {self.predation_mort_rate.shape}")
        print(f"other_mort_rate: {self.other_mort_rate.shape}")
        print(f"discard_fate: {self.discard_fate.shape}")
        print(f"discard_mortality: {self.discard_mortality.shape}")
        print(f"discards: {self.discards.shape}")

        print(f"is_detritus: {self.is_detritus.shape}")
        print(f"not_detritus: {self.not_detritus.shape}")

        print(f"det_immig: {self.det_immig.shape}")
        print(f"det_ba: {self.det_ba.shape}")
        print(f"det_import: {self.det_import.shape}")
        print(f"det_fate: {self.det_fate.shape}")
        print(f"other_mort: {self.other_mort.shape}")
        print(f"detritus_out_rates: {self.detritus_out_rates.shape}")

class EcotracerSource:

    def __init__(self, core: CoreInterface):
        # We do not store core to ensure picklability

        self.immigration_c = np.array(core.Ecotracer.get_immigration_concentrations())
        self.meta_dec_r = np.array(core.Ecotracer.get_metabolic_decay_rates())
        self.phys_dec_r = np.array(core.Ecotracer.get_physical_decay_rates())
        self.dir_abs_r = np.array(core.Ecotracer.get_direct_absorption_rates())
        self.assim_eff = 1 - np.array(core.Ecotracer.get_excretion_rates())
        self.base_inflow = core.Ecotracer.get_base_inflow_rate()
        self.env_decay = core.Ecotracer.get_env_decay_rate()
        self.env_volume_exchange_loss = core.Ecotracer.get_env_volume_exchange_loss()
