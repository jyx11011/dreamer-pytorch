from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import AtariDreamerModel


class DMCDreamerAgent(DreamerAgent):

    def __init__(self, ModelCls=AtariDreamerModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

