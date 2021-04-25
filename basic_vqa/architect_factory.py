import config
from pcdarts.architect import Architect
from pcdarts.architect_lct import ArchitectLct

def get_architect(ef_model, w_model,
        ef_optimizer, w_optimizer):
    if config.ARCH_TYPE == 'fixed':
        return None
    elif config.ARCH_TYPE == 'darts':
        if config.SKIP_STAGE2:
            return Architect( ef_model )
        else:
            return ArchitectLct( ef_model, w_model,
                    ef_optimizer, w_optimizer )
    else:
        assert False and 'unrecognized ARCH_TYPE'
