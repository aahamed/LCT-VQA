import config
from pcdarts.architect import Architect
from pcdarts.architect_lct import ArchitectLct

def get_architect(ef_model, w_model,
        ef_optimizer, w_optimizer):
    if 'fixed' in config.ARCH_TYPE:
        return None
    elif config.ARCH_TYPE == 'pcdarts':
        if config.SKIP_STAGE2:
            return Architect( ef_model )
        else:
            return ArchitectLct( ef_model, w_model,
                    ef_optimizer, w_optimizer )
    else:
        assert False and 'unrecognized ARCH_TYPE'
